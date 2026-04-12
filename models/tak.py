from copy import deepcopy
import json
import os
import numpy as np
import torch.nn as nn

from typing import Any

from utils import binary_to_boolean_type

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch.func as func
import gc
import open_clip

from typing import Tuple, List

import torch
from models.tak_utils.backbone import Backbone, create_clip
from models.tak_utils.backbone import build_classification_head
from models.tak_utils.utils import FisherLoader
from models.tak_utils.utils import OptimizerBuilder
from models.tak_utils.utils import compute_acc_on_last_task

from models.tak_utils.fisher_kfac import KFACComputer

from models.tak_utils.merging import get_merging_function
from models.tak_utils.utils import get_parameter
from utils.evaluate import evaluate

import wandb


class TAK(ContinualModel):
    """Task Arithmetic with KFAC regularization"""

    NAME = "tak"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]
    net: Backbone

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(
            optimizer="adamw",
            lr=0.0003,
            optim_wd=0.1,
        )

        clip_group = parser.add_argument_group("TAK CLIP")
        clip_group.add_argument(
            "--clip_backbone",
            type=str,
            default="ViT-B/16",
            help="Backbone architecture for CLIP",
            choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"],
        )
        clip_group.add_argument(
            "--ft_linears",
            type=binary_to_boolean_type,
            default=1,
            help="Set to 1 fine-tune linear layers",
        )
        clip_group.add_argument(
            "--ft_attention",
            type=binary_to_boolean_type,
            default=1,
            help="Set to 1 fine-tune attention layers",
        )
        clip_group.add_argument(
            "--ft_ln",
            type=binary_to_boolean_type,
            default=1,
            help="Set to 1 fine-tune layer norm",
        )
        clip_group.add_argument(
            "--ft_class_embed",
            type=binary_to_boolean_type,
            default=1,
            help="Set to 1 fine-tune class embedding layers",
        )
        clip_group.add_argument(
            "--ft_proj",
            type=binary_to_boolean_type,
            default=1,
            help="Set to 1 fine-tune projection layers",
        )
        clip_group.add_argument(
            "--ft_pos_embed",
            type=binary_to_boolean_type,
            default=0,
            help="Set to 1 fine-tune posistional embedding",
        )
        clip_group.add_argument(
            "--ft_conv",
            type=binary_to_boolean_type,
            default=0,
            help="Set to 1 fine-tune convolutional layers",
        )

        merging_group = parser.add_argument_group("TAK Merging")
        merging_group.add_argument(
            "--merging",
            type=str,
            default="ta",
            choices=["ta", "dare", "iso", "ties", "tsv"],
            help="Merging strategy for task vectors",
        )
        merging_group.add_argument(
            "--alpha_merging",
            type=str,
            default="one",
            help="Alpha used during merge."
            "NOTE: some merging strategy (ta, dare, ties) "
            "rescale the `alpha_merging` by the total number of tasks. "
            "To avoid errors, the value 'one' ensures that alpha=1 for each task",
        )

        main_group = parser.add_argument_group("TAK Main")
        main_group.add_argument(
            "--save_task_vectors",
            type=binary_to_boolean_type,
            default=0,
            help="Save computed task vectors?",
        )
        main_group.add_argument(
            "--virtual_bs_n",
            type=int,
            default=1,
            help="chose how many chunks for vitual batch size",
        )
        main_group.add_argument(
            "--default_scale_factor",
            type=float,
            default=1,
            choices=[0, 1],
            help="Default scale factor for layer scaling if a single eigenvalue is present. 0 means no scaling, 1 means full scaling.",
        )
        main_group.add_argument(
            "--reg_lambda",
            type=float,
            default=500,
            help="Regularization weight (lambda in the paper)",
        )
        main_group.add_argument(
            "--fisher_ft_proj_scaler",
            type=float,
            default=0.1,
            help="Regularization scaling coeff. for the final linear projection",
        )
        main_group.add_argument(
            "--fisher_norm_scaler",
            type=float,
            default=10,
            help="Regularization scaling coeff. for inner feed-forward layers",
        )
        main_group.add_argument(
            "--scheduler_ntk",
            type=str,
            default="cosine",
            choices=["none", "cosine", "cosine_plus", "decay", "step"],
            help="LR scheduler type",
        )
        main_group.add_argument(
            "--clip_grad_norm",
            type=float,
            default=None,
            required=False,
            help="Gradient clipping norm value - used if >0 and not None",
        )

        kfac_group = parser.add_argument_group("TAK KFAC")
        kfac_group.add_argument(
            "--load_fisher",
            type=binary_to_boolean_type,
            default=0,
            help="Load KFAC map from cache?",
        )
        kfac_group.add_argument(
            "--fisher_cache",
            type=str,
            default="fisher_cache",
            help="Path on which to save or load KFAC maps. "
            "Supports local directories, HTTP(S) base URLs, and HuggingFace sources "
            "using `hf://<owner>/<repo>/<optional/subpath>@<optional_revision>`.",
        )
        kfac_group.add_argument(
            "--train_percent",
            type=str,
            default="1.0",
            help="Percentage of training data used to compute the fisher information matrix. \
                                 If float, it represents the percentage of the training set. \
                                 If integer, it represents the number of samples used. \
                                 Put 1.0 to use the entire training set.",
        )
        kfac_group.add_argument(
            "--fisher_task_id",
            type=int,
            default=None,
            required=False,
            help="Compute KFAC approx. on this specific task",
        )
        kfac_group.add_argument(
            "--fisher_ideal",
            type=binary_to_boolean_type,
            default=0,
            help="Keep and use the fisher of each task (ideal - Eq. 7) or just the accumulated one (Eq. 8)",
        )
        kfac_group.add_argument(
            "--fisher_num_samples_expectation",
            type=int,
            default=1,
            help="Compute KFAC approx. on a fixed number of samples, subset of all the task",
        )

        ablation_group = parser.add_argument_group("TAK Ablation")
        ablation_group.add_argument(
            "--tangent",
            type=binary_to_boolean_type,
            default=1,
            help="Use or disable linearized training and inference (NTK regime)",
        )

        extra_group = parser.add_argument_group("TAK Extra")
        extra_group.add_argument("--use_lora", type=binary_to_boolean_type, default=0)
        extra_group.add_argument(
            "--fp_precision",
            type=str,
            default="fp32",
            choices=["fp32", "fp64"],
            help="Floating point fp_precision used during KFAC computations",
        )
        extra_group.add_argument(
            "--resume",
            type=binary_to_boolean_type,
            default=0,
            help="Resume previous training? NOTE: requires `load_path`",
        )
        extra_group.add_argument(
            "--load_path",
            type=str,
            default=None,
            required=False,
            help="Path from which load the previous task's task vectors. Used with `resume=1`",
        )

        eval_group = parser.add_argument_group("TAK Evaluation")
        eval_group.add_argument(
            "--alpha_sweep_start",
            type=float,
            default=0.1,
            help="Starting merging alpha value for sensitivity analysis - used by `compute_metrics_by_alpha`",
        )
        eval_group.add_argument(
            "--alpha_sweep_end",
            type=float,
            default=1.5,
            help="Final merging alpha value for sensitivity analysis - used by `compute_metrics_by_alpha`",
        )
        eval_group.add_argument(
            "--alpha_sweep_step",
            type=float,
            default=0.1,
            help="Step alpha value for sensitivity analysis - used by `compute_metrics_by_alpha`",
        )

        return parser

    def _parse_train_percent(self):
        if not isinstance(self.args.train_percent, str):
            return self.args
        raw = self.args.train_percent.strip()
        if ("." in raw) or ("e" in raw) or ("E" in raw):
            val = float(raw)
            assert 0 < val <= 1, "If float, train_percent must be in (0,1]"
            self.args.train_percent = val
        else:
            val = int(raw)
            assert val >= 1, "If integer, train_percent must be >= 1"
            self.args.train_percent = val
        return self.args

    def _ensure_merging_alpha(self):
        if self.args.alpha_merging == "one":
            if self.args.merging in ("ta", "dare", "ties"):
                self.args.alpha_merging = self.dataset.N_TASKS
        else:
            try:
                self.args.alpha_merging = float(self.args.alpha_merging)
            except ValueError:
                raise ValueError("alpha_merging must be 'one' or a float value")

    def __init__(self, backbone, loss, args, transform, dataset):
        assert dataset is not None

        _, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            args.clip_backbone, pretrained="openai", device=torch.device("cpu")
        )

        clip_model = create_clip(args.clip_backbone, torch.device(args.device))

        super().__init__(clip_model, loss, args, transform, dataset=dataset)
        self._ensure_merging_alpha()
        self._parse_train_percent()

        if self.args.resume:
            assert self.args.load_path is not None, (
                "Must provide load_path when resume is set to True"
            )

        if self.args.save_task_vectors:
            os.makedirs(self.args.checkpoint_path, exist_ok=True)
            tv_path = os.path.join(
                self.args.checkpoint_path,
                f"{self.args.conf_jobnum}_{dataset.NAME}_args.json",
            )
            with open(tv_path, "w") as f:
                json_args = deepcopy(vars(args))
                del json_args[
                    "device"
                ]  # device not serializable because torch people are dumb
                json.dump(json_args, f)

        self.net = Backbone(clip_model, dataset, args)
        self.param_names = [
            name for name, _ in self.net.visual_encoder.named_parameters()
        ]

        for name, param in self.net.named_parameters():
            param.requires_grad = False

        torch.backends.cuda.enable_mem_efficient_sdp(False)

        clip_model = clip_model.to(dtype=torch.float32)
        clip_model.eval()

        self.clip_model = clip_model
        self.clip_transform = train_preprocess
        self.clip_eval_transform = val_preprocess

        self.fisher_computer = KFACComputer(
            self.device,
            self.args.debug_mode == 1,
            train_percent=self.args.train_percent,
            num_samples_expectation=self.args.fisher_num_samples_expectation,
            fp_precision=self.args.fp_precision,
        )

        self.fisher_loader = FisherLoader(
            self.args.fisher_cache, dataset.NAME, self.device, fp_precision="fp32"
        )

        self.optimizer_builder = OptimizerBuilder(cmd_args=self.args)

        self.cur_offset = None
        self.cls_head: nn.Module = None

        self.delta_w_dict: dict[str, Any] = None
        self.delta_w_names: list[str] = None
        self.delta_w_shapes: dict[str, Any] = None

        self.scheduler1 = None

        self.tasks_ggT = {}
        self.tasks_ffT = {}
        self.tasks_aaT = {}

        self.coeffs = []
        self.layer_scale_factors: dict[str, float] = {}

        self.num_batches = 0

        self.merging = get_merging_function(self.args, self.device)

        self.merged_task_vector = []

        self.num_total_tasks: int = dataset.N_TASKS
        self.dataset_name: int = dataset.NAME

        self.individual_acc, self.individual_mask_acc = [], []
        self.norm_acc, self.norm_mask_acc = [], []

        self.task_loaded = False

    def create_param_like(self, param, requires_grad):
        return [
            torch.zeros_like(
                param,
                dtype=torch.float32,
                requires_grad=requires_grad,
                device=self.args.device,
            )
        ]

    def create_lora_param_like(self, fin, fout, requires_grad, r1=None, r2=None):
        r1 = 16 if r1 is None else r1
        r2 = 16 if r2 is None else r2
        config = ("kaiming", "zeros")
        return get_parameter(
            (fout, r2), self.device, config[1], False, requires_grad
        ), get_parameter((r1, fin), self.device, config[0], False, requires_grad)

    def begin_task(self, dataset):
        torch.cuda.empty_cache()

        dataset.test_loaders[-1].dataset.transform = self.clip_eval_transform
        dataset.train_loader.dataset.transform = self.clip_transform

        self.cur_offset = self.get_offsets(self.current_task)

        if isinstance(dataset.N_CLASSES_PER_TASK, int):
            self.cpt = dataset.N_CLASSES_PER_TASK
        else:
            self.cpt = dataset.N_CLASSES_PER_TASK[-1]

        if self.current_task != 0:
            self.net.task_id += 1

        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset
        )

        print("\nRELOADING CLIP VISUAL ENCODER")
        self.net.copy_visual_encoder(self.clip_model)

        for param in self.net.visual_encoder.parameters():
            param.requires_grad = False

        print("\nCLIP VISUAL ENCODER RELOADED\n\n")

        self.delta_w_dict = {}
        self.delta_w_shapes = {}

        for name, param in self.net.visual_encoder.named_parameters():
            self.delta_w_shapes[name] = param.shape

            if self.args.use_lora == 1 and len(param.shape) == 2:
                fout, fin = param.shape[0], param.shape[1]

                if "mlp" in name:
                    B, A = self.create_lora_param_like(
                        fin, fout, self.args.ft_linears == 1
                    )
                    self.delta_w_dict[name] = [B, A]
                elif "attn" in name:
                    B, A = self.create_lora_param_like(
                        fin, fout, self.args.ft_attention == 1, r1=16 * 3, r2=16 * 3
                    )
                    self.delta_w_dict[name] = [B, A]
                elif "proj" in name:
                    if name == "proj":
                        # skip, this is the projection layer of the visual encoder which has beeen replaced
                        continue
                    B, A = self.create_lora_param_like(
                        fin, fout, self.args.ft_proj == 1
                    )
                    self.delta_w_dict[name] = [B, A]
            else:
                if "mlp" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_linears == 1
                    )
                elif "attn" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_attention == 1
                    )
                elif "proj" in name:
                    if name == "proj":
                        # skip, this is the projection layer of the visual encoder which has beeen replaced
                        continue
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_proj == 1
                    )
                elif "ln" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_ln == 1
                    )
                elif "class_embed" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_class_embed == 1
                    )
                elif "conv" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_conv == 1
                    )
                elif "positional_embedding" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_pos_embed == 1
                    )

        self.delta_w_names = list(self.delta_w_dict.keys())

        if not self.args.load_fisher:
            if (
                self.args.fisher_task_id is None
                or self.current_task == self.args.fisher_task_id
            ):
                dataset.train_loader.dataset.transform = self.clip_eval_transform

                ggT, aaT, ffT, num_ggT, num_aaT = self.fisher_computer.compute(
                    self.net, None, self.delta_w_names, dataset, use_head=False
                )

                dataset.train_loader.dataset.transform = self.clip_transform

                self.fisher_loader.store_kfac(
                    self.current_task, ggT, aaT, ffT, num_ggT, num_aaT
                )
            else:
                print(
                    f"Skipping Fisher computation for task {self.current_task} as "
                    f"it is not the specified task {self.args.fisher_task_id}."
                )
        else:
            counts = [
                self.fisher_loader.load_kfac(t, only_counts=True)
                for t in range(dataset.N_TASKS)
            ]
            tot_ggT = sum(
                [
                    cnt[0]
                    for idx_cnt, cnt in enumerate(counts)
                    if idx_cnt != self.current_task
                ]
            )
            tot_aaT = sum(
                [
                    cnt[1]
                    for idx_cnt, cnt in enumerate(counts)
                    if idx_cnt != self.current_task
                ]
            )

            assert tot_ggT == tot_aaT

            self.coeffs = []
            num_penalties = dataset.N_TASKS - 1 if self.args.fisher_ideal else 1

            for t in range(dataset.N_TASKS):
                ggT, aaT, ffT, cur_num_ggT, cur_num_aaT = self.fisher_loader.load_kfac(
                    t
                )
                assert cur_num_ggT == cur_num_aaT
                coeff = cur_num_ggT / tot_ggT

                if t == 0:
                    for key in aaT.keys():
                        if key in self.tasks_ggT.keys():
                            for p_l in range(num_penalties):
                                self.tasks_aaT[key][p_l].zero_()
                                self.tasks_ggT[key][p_l].zero_()
                        else:
                            self.tasks_aaT[key] = [
                                torch.zeros_like(aaT[key]) for _ in range(num_penalties)
                            ]
                            self.tasks_ggT[key] = [
                                torch.zeros_like(ggT[key]) for _ in range(num_penalties)
                            ]

                    for key in ffT.keys():
                        if key in self.tasks_ffT.keys():
                            self.tasks_ffT[key].zero_()
                        else:
                            self.tasks_ffT[key] = torch.zeros_like(ffT[key])

                if t != self.current_task:
                    self.coeffs.append(coeff)

                    for key in ffT.keys():
                        self.tasks_ffT[key].add_(ffT[key] / tot_ggT)

                    for key in aaT.keys():
                        aaT[key].div_(cur_num_aaT)
                        ggT[key].div_(cur_num_ggT)

                        if self.args.fisher_ideal == 0:
                            self.tasks_aaT[key][0].add_(
                                (cur_num_aaT / tot_aaT) * aaT[key]
                            )
                            self.tasks_ggT[key][0].add_(ggT[key])
                        else:
                            t_hat = t if t <= self.current_task else t - 1
                            self.tasks_ggT[key][t_hat].copy_(ggT[key])
                            self.tasks_aaT[key][t_hat].copy_(aaT[key])

                del aaT, ggT, ffT

        all_params = [
            p for param_list in self.delta_w_dict.values() for p in param_list
        ]
        num_batches: int = len(dataset.train_loader)

        self.opt, self.scheduler1 = self.optimizer_builder.build_opt_and_sched(
            all_params, num_batches
        )

        if self.args.resume:
            self.task_loaded = self.load_task_vectors()
            if self.task_loaded:
                print(f"Task vectors for {self.current_task} loaded successfully")
                self.args.n_epochs = 0
                self.n_epochs = 0

        self.train()

    def get_parameter_from_dict(self, name):
        assert name in self.delta_w_names
        list_params = self.delta_w_dict[name]
        if len(list_params) == 1:
            return list_params[0]
        elif len(list_params) == 2:
            return list_params[0] @ list_params[1]
        else:
            raise ValueError

    def get_all_parameters_from_dict(self):
        return [self.get_parameter_from_dict(k) for k in self.delta_w_names]

    def end_task(
        self, dataset: ContinualDataset
    ) -> None:  # TODO  set the model in eval mode
        print(f"Current task: {self.current_task}")

        self.eval()

        self.merged_task_vector = []

        for i, key in enumerate(self.delta_w_names):
            self.merged_task_vector.append(
                torch.clone(self.get_parameter_from_dict(key))
            )

        actual_seen_classes = self.n_seen_classes

        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset, all_heads=True
        )
        self._n_seen_classes = dataset.N_CLASSES

        acc, acc_mask_classes = compute_acc_on_last_task(self, dataset)
        self.individual_acc.append(acc)
        self.individual_mask_acc.append(acc_mask_classes)

        self._n_seen_classes = actual_seen_classes

        if self.args.save_task_vectors:
            save_data = []
            for key in self.delta_w_names:
                param_to_save = self.get_parameter_from_dict(key)

                if isinstance(param_to_save, list):
                    save_data.append([p.clone().cpu() for p in param_to_save])
                else:
                    save_data.append(param_to_save.clone().cpu())

            base_path = os.path.join(
                self.args.checkpoint_path,
                self.args.fisher_cache,
                f"{self.args.conf_jobnum}_{dataset.NAME}_task_{self.current_task}",
            )

            tv_path = base_path + ".pt"
            os.makedirs(os.path.dirname(tv_path), exist_ok=True)

            torch.save(save_data, tv_path)
            torch.save(self.cls_head.state_dict(), base_path + "_cls_head.pt")
            torch.save({"delta_w_names": self.delta_w_names}, base_path + "_meta.pt")
            print(f"Task vector saved to {tv_path}")

        del self.merged_task_vector[:]
        del self.merged_task_vector

        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset, eval=True
        )

        for i, key in enumerate(self.delta_w_names):
            num_params = len(self.delta_w_dict[key])
            for p_l in range(num_params):
                self.delta_w_dict[key][p_l].requires_grad = False

        self.merging.add(
            {key: self.get_parameter_from_dict(key) for key in self.delta_w_names}
        )

        self.merged_task_vector = self.merging.merge(self.delta_w_names)

        self.opt.zero_grad()
        self.opt = None

        self.net.copy_visual_encoder(self.clip_model)

        torch.cuda.empty_cache()

        del self.opt, self.scheduler1, self.delta_w_dict
        gc.collect()

        return super().end_task(dataset)

    def end_eval(self, dataset: ContinualDataset, accs: Tuple[List, List]) -> None:
        def safe_den(y, eps=1e-8):
            return y if abs(y) >= eps else y + eps

        self.norm_acc = [
            acc / safe_den(self.individual_acc[t]) for t, acc in enumerate(accs[0])
        ]
        self.norm_mask_acc = [
            acc / safe_den(self.individual_mask_acc[t]) for t, acc in enumerate(accs[1])
        ]

        if self.args.nowand == 0:
            wandb.log(
                {
                    "RESULT_mean_norm_acc": sum(self.norm_acc) / len(self.norm_acc),
                    "RESULT_mean_norm_mask_acc": sum(self.norm_mask_acc)
                    / len(self.norm_mask_acc),
                    "Task": self.current_task,
                }
            )
        if self.current_task == self.num_total_tasks-1:
            self.compute_metrics_by_alpha()

    def penalty_weight(self):
        loss_reg, loss_reg_ffT, loss_ft_proj = 0, 0, 0
        loss_reg_cls_emb = 0

        for name in self.delta_w_names:
            if name in self.tasks_aaT.keys():
                delta_W = self.get_parameter_from_dict(name)
                bias_name = name.replace("weight", "bias")

                if name.replace("weight", "bias") in self.delta_w_names:
                    assert "weight" in name
                    delta_bias = self.get_parameter_from_dict(bias_name)
                    delta_W = torch.cat((delta_W, delta_bias.unsqueeze(1)), 1)

                for task_id in range(len(self.tasks_aaT[name])):
                    aaT_past = self.tasks_aaT[name][task_id]
                    ggT_past = self.tasks_ggT[name][task_id]

                    norm_coeff = self.coeffs[task_id] if self.args.fisher_ideal else 1
                    loss_ = torch.trace(ggT_past @ delta_W @ aaT_past @ delta_W.T)

                    if name == "lin_proj.weight":
                        loss_ft_proj += norm_coeff * loss_
                    else:
                        loss_reg += norm_coeff * loss_

            if name in self.tasks_ffT.keys():
                delta_W = self.get_parameter_from_dict(name).unsqueeze(0)
                ffT_past = self.tasks_ffT[name]
                reg_w = torch.trace(delta_W @ ffT_past @ delta_W.T)
                if "class_embedding" in name:
                    loss_reg_cls_emb += reg_w
                else:
                    loss_reg_ffT += reg_w

        return loss_reg, loss_reg_ffT, loss_ft_proj, loss_reg_cls_emb

    def create_functional(self, inputs, delta_names):
        def func_network(param_values):
            param = {name: param for name, param in zip(delta_names, param_values)}
            features = func.functional_call(self.net.visual_encoder, param, inputs)
            return nn.functional.normalize(features, dim=-1)

        return func_network

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.args.resume and self.task_loaded:
            return 0.0

        if self.args.tangent:
            forward_fun = self.create_functional(inputs, self.delta_w_names)
            params = [
                param
                for name, param in self.net.visual_encoder.named_parameters()
                if name in self.delta_w_names
            ]
            image_features, jvp = func.jvp(
                forward_fun,
                (tuple(params),),
                (tuple(self.get_all_parameters_from_dict()),),
            )
            image_features = image_features + jvp

        else:
            tunable_params = [
                p
                for n, p in self.net.visual_encoder.named_parameters()
                if n in self.delta_w_names
            ]
            dict_param = {
                name: param + net_param
                for name, param, net_param in zip(
                    self.delta_w_names,
                    self.get_all_parameters_from_dict(),
                    tunable_params,
                )
            }

            image_features = func.functional_call(
                self.net.visual_encoder, dict_param, inputs
            )
            image_features = nn.functional.normalize(image_features, dim=-1)

        similarity = self.cls_head(image_features)
        loss_task = self.loss(similarity, labels - self.n_past_classes)
        loss = loss_task / self.args.virtual_bs_n
        loss.backward()

        if (
            (self.args.load_fisher)
            and (self.task_iteration > 0)
            and (self.task_iteration % self.args.virtual_bs_n == 0)
        ):
            loss_penalty, loss_reg_ffT, loss_ft_proj, loss_reg_cls_emb = (
                self.penalty_weight()
            )
            loss_reg = (
                self.args.reg_lambda * loss_penalty
                + self.args.reg_lambda * self.args.fisher_norm_scaler * loss_reg_ffT
                + self.args.reg_lambda * self.args.fisher_ft_proj_scaler * loss_ft_proj
                + self.args.reg_lambda * self.args.fisher_norm_scaler * loss_reg_cls_emb
            )

            loss_reg.backward()

        if (
            self.task_iteration > 0
        ) and self.task_iteration % self.args.virtual_bs_n == 0:
            if self.scheduler1:
                self.scheduler1(self.task_iteration // self.args.virtual_bs_n)
            if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for group in self.opt.param_groups for p in group["params"]),
                    self.args.clip_grad_norm,
                )
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()

    @torch.no_grad()
    def forward(self, x):
        if self.args.tangent:
            forward_fun = self.create_functional(x, self.delta_w_names)
            params = [
                param
                for name, param in self.net.visual_encoder.named_parameters()
                if name in self.delta_w_names
            ]
            image_features, jvp = func.jvp(
                forward_fun,
                (tuple(params),),
                (tuple(self.merged_task_vector),),
            )
            image_features = image_features + jvp
        else:
            tunable_params = {
                n: p
                for n, p in self.net.visual_encoder.named_parameters()
                if n in self.delta_w_names
            }

            dict_param = {}
            for i, key in enumerate(self.delta_w_names):
                dict_param[key] = tunable_params[key] + self.merged_task_vector[i]

            image_features = func.functional_call(
                self.net.visual_encoder, dict_param, x
            )
            image_features = nn.functional.normalize(image_features, dim=-1)

        similarity = self.cls_head(image_features)
        return similarity[:, : self.n_seen_classes]

    def get_debug_iters(self):
        return 5

    def load_task_vectors(self):
        """
        Returns:
            bool: True if loading was successful, False otherwise
        """

        print(
            f"Loading task vector number: {self.current_task} from: {self.args.load_path}"
        )

        tv_path = self.args.load_path.replace(
            "_args.json", f"_task_{self.current_task}.pt"
        )

        if os.path.exists(tv_path):
            print(f"Found task vector {self.current_task}: {tv_path}")
        else:
            print(f"WARNING: Missing task vector for task {self.current_task}")
            return False

        # Load task vectors
        task_vectors = torch.load(tv_path, map_location=self.device)

        # Load metadata from the last task's task vector
        meta_path = tv_path.replace(".pt", "_meta.pt")
        if os.path.exists(meta_path):
            meta_data = torch.load(meta_path, map_location=self.device)
            delta_w_names = meta_data["delta_w_names"]
            print(f"Loaded metadata with {len(self.delta_w_names)} parameters")
        else:
            print("WARNING: Metadata file not found")
            return False
        # self.delta_w_dict = {name: param for name, param in zip(self.delta_w_names, task_vectors)}
        for name, param in zip(delta_w_names, task_vectors):
            self.delta_w_dict[name] = [param.clone().detach().to(self.device)]

        print(f"Added task vector {self.current_task} to merging")

        return True

    def compute_metrics_by_alpha(self):
        def safe_den(y, eps=1e-8):
            return y if abs(y) >= eps else y + eps

        if not hasattr(self, "_alpha_sweep_next_id"):
            self._alpha_sweep_next_id = 1
        sweep_id = self._alpha_sweep_next_id
        self._alpha_sweep_next_id += 1
        if sweep_id == 1:
            metric_name = "alpha_sweep"
            norm_metric_name = "norm_alpha_sweep"
        else:
            metric_name = f"alpha_sweep_{sweep_id}"
            norm_metric_name = f"norm_alpha_sweep_{sweep_id}"
        if self.args.nowand == 0:
            if not hasattr(self, "_alpha_metric_defined"):
                wandb.define_metric("alpha")
                self._alpha_metric_defined = True
            wandb.define_metric(metric_name, step_metric="alpha")
            wandb.define_metric(norm_metric_name, step_metric="alpha")

        alphas = np.arange(
            self.args.alpha_sweep_start,
            self.args.alpha_sweep_end + self.args.alpha_sweep_step,
            self.args.alpha_sweep_step,
        ).tolist()
        alphas = [a * self.num_total_tasks for a in alphas]

        # alpha sweep
        for alpha in alphas:
            self.merging.set_alpha(alpha)
            self.merged_task_vector = self.merging.merge(self.delta_w_names)
            accs, accs_mask_classes = evaluate(self, self.dataset)
            norm_mask_acc = [
                acc / safe_den(self.individual_mask_acc[t])
                for t, acc in enumerate(accs_mask_classes)
            ]
            print(
                f"Alpha: {alpha} - Acc: {sum(accs_mask_classes) / len(accs_mask_classes):.4f} - Norm Acc: {sum(norm_mask_acc) / len(norm_mask_acc):.4f}"
            )
            print(f"single tasks accs: {accs_mask_classes}")
            if self.args.nowand == 0:
                wandb.log(
                    {
                        "alpha": alpha,
                        metric_name: sum(accs_mask_classes) / len(accs_mask_classes),
                        norm_metric_name: sum(norm_mask_acc) / len(norm_mask_acc),
                    }
                )
