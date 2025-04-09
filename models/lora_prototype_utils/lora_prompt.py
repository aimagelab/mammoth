from typing import TYPE_CHECKING
from tqdm import tqdm
from functools import partial

import torch

from utils.conf import get_device

from models.lora_prototype_utils.fisher import UnbiasedFisherModule, AugmentedFisherModule
from models.lora_prototype_utils.utils import IncrementalClassifier
from models.lora_prototype_utils.lora_vit import VisionTransformer as LoRAViT

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone


def get_fisher_caller(args):
    if args.use_iel:
        fisher_caller = partial(UnbiasedFisherModule, beta_iel=args.beta_iel)
    else:
        fisher_caller = partial(AugmentedFisherModule, beta_iel=args.beta_iel,
                                alpha_ita=args.alpha_ita)

    return fisher_caller


class Model(torch.nn.Module):

    def __init__(self, args, seq_dataset: 'ContinualDataset', backbone: 'MammothBackbone'):

        super().__init__()

        self.compute_fisher = self.__compute_fisher_hooks

        self.num_classes = seq_dataset.N_CLASSES
        self.num_tasks = seq_dataset.N_TASKS

        self.current_task = 0
        self.class_offsets = [seq_dataset.get_offsets(t) for t in range(self.num_tasks)]
        self.num_classes_per_task = [e - s for (s, e) in self.class_offsets]

        self.device = get_device()

        self.vit = LoRAViT.from_mammoth(backbone)
        self.vit.head = IncrementalClassifier(self.vit.embed_dim, self.num_classes_per_task[0]).to(self.device)

        self.output_dim = self.vit.embed_dim
        self.embed_dim = self.vit.embed_dim
        self.mlp_ratio = self.vit.mlp_ratio
        self.tuning_style = args.tuning_style

        self.tuner = self.init_tuner(args, seq_dataset)

        param_lored, param_resolution_dict = self.get_params_to_tune()

        self.fisher_mc_classes = args.fisher_mc_classes
        self.req_weight_cls = args.req_weight_cls
        self.simple_reg_weight_cls = args.simple_reg_weight_cls

        fisher_caller = get_fisher_caller(args)

        self.fisher_dict = torch.nn.ModuleDict({
            param_resolution_dict[n]: fisher_caller(p)
            for n, p in param_lored.items()
        })

        self.fisher_dict_cls = torch.nn.ModuleDict({
            n: UnbiasedFisherModule(p, args.req_weight_cls)
            for n, p in self.vit.head.heads[self.current_task].named_parameters()
        })

        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.iteration = 0

    def set_current_task(self, task_id):
        self.current_task = task_id
        self.tuner.set_current_task(task_id)

        if self.current_task > 0:
            device = self.fisher_dict_cls.weight.unnormalized_fisher.device
            self.fisher_dict_cls.update({
                n: UnbiasedFisherModule(p, self.req_weight_cls).to(device)
                for n, p in self.vit.head.heads[self.current_task].named_parameters()
            })

    def init_tuner(self, args, seq_dataset):
        if args.tuning_style == 'lora':
            from models.lora_prototype_utils.tuners.lora_tuner import LoRATuner
            return LoRATuner(args, self.device, seq_dataset,
                             embed_dim=self.embed_dim,
                             mlp_ratio=self.mlp_ratio)
        elif args.tuning_style == 'full':
            from models.lora_prototype_utils.tuners.full_tuner import FullTuner
            return FullTuner(args, self.device, seq_dataset,
                             embed_dim=self.embed_dim,
                             mlp_ratio=self.mlp_ratio)
        elif args.tuning_style == 'ia3':
            from models.lora_prototype_utils.tuners.ia3_tuner import IA3Tuner
            return IA3Tuner(args, self.device, seq_dataset,
                            embed_dim=self.embed_dim,
                            mlp_ratio=self.mlp_ratio,
                            orig_model=self.vit)
        else:
            raise ValueError(f"Unknown tuning style: {args.tuning_style}")

    def build_optimizer_args(self, lr_params, lr_classifier=None,
                             wd_params: float = 0, wd_classifier: float = 0):

        lora_vars = self.tuner.get_current_optimizing_parameters()
        lr_classifier = lr_classifier if lr_classifier is not None else lr_params

        lora_params = {
            'params': lora_vars,
            'lr': lr_params,
            'weight_decay': wd_params
        }

        base_fc_params = {
            'params': [p for p in self.vit.head.parameters() if p.requires_grad],
            'lr': lr_classifier,
            'weight_decay': wd_classifier
        }

        return [lora_params, base_fc_params]

    def get_params_to_tune(self):
        paramname_lored = set()
        param_resolution_dict = dict()

        expansion = {
            'qkv': 'attn.qkv',
            'proj': 'attn.proj',
            'fc1': 'mlp.fc1',
            'fc2': 'mlp.fc2'
        }

        for layer_idx, vars in self.tuner.get_params_to_tune().items():
            for v in vars:
                key_param_name = f'blocks.{layer_idx}.{expansion[v]}.weight'
                paramname_lored.add(key_param_name)
                param_resolution_dict[key_param_name] = f'{v}_{layer_idx}'

        param_lored = {n: p for n, p in self.vit.named_parameters() if n in paramname_lored}

        return param_lored, param_resolution_dict

    def train(self, mode=True):
        super().train(False)
        self.tuner.train(False)
        self.vit.train(mode)
        return self

    def __compute_fisher_hooks(self, param_lored, param_resolution_dict,
                               param_cls, param_resolution_dict_cls,
                               dataset, debug_mode: bool = False):

        all_param_lored_names = list(param_resolution_dict.keys()) + list(param_resolution_dict_cls.keys())

        def to_be_fishered(name):
            if f"{name}.weight" in all_param_lored_names or f"{name}.bias" in all_param_lored_names:
                return True
            else:
                return False

        def hook_backward(module, _, grad_output):
            grad_out = grad_output[0]
            inputs = module.inputs
            if len(grad_out.shape) > 2:
                grad_out = grad_out.reshape(-1, grad_out.shape[-1])
                inputs = inputs.reshape(-1, inputs.shape[-1])
                grad_weight = (grad_output[0].permute(0, 2, 1) @ module.inputs).pow(2).sum(0)
            else:
                grad_weight = grad_out.T.pow(2) @ inputs.pow(2)

            if hasattr(module, "bias") and module.__compute_bias:
                grad_bias = grad_out.T.pow(2).sum(-1)
                if not hasattr(module, "fisher_bias"):
                    setattr(module, "fisher_bias", grad_bias)
                else:
                    module.fisher_bias += grad_bias

            if not hasattr(module, "fisher_weight"):
                setattr(module, "fisher_weight", grad_weight)
            else:
                module.fisher_weight += grad_weight

        def hook_forward(module, inputs, _):
            setattr(module, "inputs", inputs[0])

        # insert hooks
        for name, module in self.vit.named_modules():
            if to_be_fishered(name):
                if f"{name}.bias" in all_param_lored_names:
                    module.__compute_bias = True
                else:
                    module.__compute_bias = False
                module.backward_handle = module.register_full_backward_hook(hook_backward)
                module.forward_handle = module.register_forward_hook(hook_forward)

        require_grads_list = []
        for param in self.vit.parameters():
            require_grads_list.append(param.requires_grad)
            param.requires_grad = False

        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)
        for j, (examples, _, _) in tqdm(enumerate(dataset.train_loader),
                                        total=len(dataset.train_loader), desc='FISHER computation'):
            if debug_mode and j > 5:
                break
            examples = examples.to(self.device)
            num_of_examples += examples.shape[0]
            probs = torch.softmax(self.vit(examples * fake_param), dim=1)
            detached_probs = probs.detach()
            log_probs = torch.log(probs)
            fisher_sqrt = (detached_probs.sqrt() * log_probs).sum(0)
            if self.fisher_mc_classes < detached_probs.shape[1]:
                _, class_indices = torch.topk(detached_probs, self.fisher_mc_classes, dim=1)
                unique, counts = class_indices.unique(return_counts=True)
                unique = unique[torch.argsort(-counts)][:self.fisher_mc_classes]
                for i in unique:
                    fisher_sqrt[i].backward(
                        retain_graph=True if (i != unique[-1]) else False
                    )
            else:
                for i, fish in enumerate(fisher_sqrt):
                    fish.backward(
                        retain_graph=True if (i < fisher_sqrt.shape[0] - 1) else False
                    )

        # remove hooks
        for name, module in self.vit.named_modules():
            if to_be_fishered(name):
                module.backward_handle.remove()
                module.forward_handle.remove()
                module.inputs = None

        fisher = {}
        for (name, module) in self.vit.named_modules():
            for typ in ["weight", "bias"]:
                if f"{name}.{typ}" in param_resolution_dict:
                    fisher[param_resolution_dict[f"{name}.{typ}"]] = getattr(module, f"fisher_{typ}")
                    setattr(module, f"fisher_{typ}", 0)

        fisher_cls = {}
        for (name, module) in self.vit.named_modules():
            for typ in ["weight", "bias"]:
                if f"{name}.{typ}" in param_resolution_dict_cls:
                    fisher_cls[param_resolution_dict_cls[f"{name}.{typ}"]] = getattr(module, f"fisher_{typ}")
                    setattr(module, f"fisher_{typ}", 0)

        for param, req_grad in zip(self.vit.parameters(), require_grads_list):
            param.requires_grad = req_grad

        return fisher, fisher_cls, num_of_examples

    def compute_fisher_cls_generative_hooks(self, _, param_resolution_dict_cls, generative_dataloader, debug_mode: bool = False):

        all_param_lored_names = list(param_resolution_dict_cls.keys())

        def to_be_fishered(name):
            if f"{name}.weight" in all_param_lored_names or f"{name}.bias" in all_param_lored_names:
                return True
            else:
                return False

        def hook_backward(module, _, grad_output):
            grad_out = grad_output[0]
            inputs = module.inputs
            if len(grad_out.shape) > 2:
                grad_out = grad_out.reshape(-1, grad_out.shape[-1])
                inputs = inputs.reshape(-1, inputs.shape[-1])
                grad_weight = (grad_output[0].permute(0, 2, 1) @ module.inputs).pow(2).sum(0)
            else:
                grad_weight = grad_out.T.pow(2) @ inputs.pow(2)

            if hasattr(module, "bias") and module.__compute_bias:
                grad_bias = grad_out.T.pow(2).sum(-1)
                if not hasattr(module, "fisher_bias"):
                    setattr(module, "fisher_bias", grad_bias)
                else:
                    module.fisher_bias += grad_bias

            if not hasattr(module, "fisher_weight"):
                setattr(module, "fisher_weight", grad_weight)
            else:
                module.fisher_weight += grad_weight

        def hook_forward(module, inputs, _):
            setattr(module, "inputs", inputs[0])

        # insert hooks
        for name, module in self.vit.named_modules():
            if to_be_fishered(name):
                if f"{name}.bias" in all_param_lored_names:
                    module.__compute_bias = True
                else:
                    module.__compute_bias = False
                module.backward_handle = module.register_full_backward_hook(hook_backward)
                module.forward_handle = module.register_forward_hook(hook_forward)

        require_grads_list = []
        for param in self.vit.parameters():
            require_grads_list.append(param.requires_grad)
            param.requires_grad = False

        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)
        for j, (examples, _) in tqdm(enumerate(generative_dataloader),
                                     total=len(generative_dataloader), desc='FISHER computation'):
            if debug_mode and j > 5:
                break
            examples = examples.to(self.device)
            num_of_examples += examples.shape[0]
            probs = torch.softmax(self.vit.head(examples * fake_param), dim=1)
            detached_probs = probs.detach()
            log_probs = torch.log(probs)
            fisher_sqrt = (detached_probs.sqrt() * log_probs).sum(0)

            for i, fish in enumerate(fisher_sqrt):
                fish.backward(
                    retain_graph=True if (i < fisher_sqrt.shape[0] - 1) else False
                )

        # remove hooks
        for name, module in self.vit.named_modules():
            if to_be_fishered(name):
                module.backward_handle.remove()
                module.forward_handle.remove()
                module.inputs = None

        fisher_cls = {}
        for (name, module) in self.vit.named_modules():
            for typ in ["weight", "bias"]:
                if f"{name}.{typ}" in param_resolution_dict_cls:
                    fisher_cls[param_resolution_dict_cls[f"{name}.{typ}"]] = getattr(module, f"fisher_{typ}")
                    setattr(module, f"fisher_{typ}", 0)

        for param, req_grad in zip(self.vit.parameters(), require_grads_list):
            param.requires_grad = req_grad

        return fisher_cls, num_of_examples

    def get_params_of_classifier(self):
        param_resolution_dict_cls = {
            f'head.heads.{self.current_task}.weight': 'weight',
            f'head.heads.{self.current_task}.bias': 'bias',
        }
        param_cls = {
            n: p for n, p in self.vit.named_parameters()
            if p.requires_grad and n in param_resolution_dict_cls.keys()
        }
        return param_cls, param_resolution_dict_cls

    def update_fisher(self, dataset, generative_dataloader=None, debug_mode: bool = False):

        param_lored, param_resolution_dict = self.get_params_to_tune()
        param_cls, param_resolution_dict_cls = self.get_params_of_classifier()

        fisher, fisher_cls, num_of_examples = \
            self.compute_fisher(param_lored, param_resolution_dict,
                                param_cls, param_resolution_dict_cls, dataset, debug_mode)

        num_of_examples_cls = num_of_examples
        for cnt, namevar in enumerate(param_cls.keys()):
            namevar_mapped = param_resolution_dict_cls[namevar]
            fisher_cls[namevar_mapped].zero_()

        if generative_dataloader is not None:
            fisher_cls, num_of_examples_cls = self.compute_fisher_cls_generative_hooks(param_cls, param_resolution_dict_cls, generative_dataloader, debug_mode)

        with torch.no_grad():

            sum_elem, sum_trace = 0, 0

            for cnt, namevar in enumerate(param_lored.keys()):
                namevar_mapped = param_resolution_dict[namevar]

                self.fisher_dict[namevar_mapped].update(fisher[namevar_mapped],
                                                        num_examples=num_of_examples)

                sum_elem += self.fisher_dict[namevar_mapped].get_num_elems()
                sum_trace += self.fisher_dict[namevar_mapped].trace()

            sum_elem_cls, sum_trace_cls = 0, 0

            for cnt, namevar in enumerate(param_cls.keys()):
                namevar_mapped = param_resolution_dict_cls[namevar]

                self.fisher_dict_cls[namevar_mapped].update(fisher_cls[namevar_mapped],
                                                            num_examples=num_of_examples_cls)

                sum_elem_cls += self.fisher_dict_cls[namevar_mapped].get_num_elems()
                sum_trace_cls += self.fisher_dict_cls[namevar_mapped].trace()

        del fisher
        del fisher_cls

    def compute_classifier_reg_loss(self, cls_ref: torch.nn.Module,
                                    do_backward: bool):

        with torch.set_grad_enabled(do_backward):
            tau_weight = (self.vit.head.heads[self.current_task].weight -
                          cls_ref.heads[self.current_task].weight.detach())
            tau_bias = (self.vit.head.heads[self.current_task].bias -
                        cls_ref.heads[self.current_task].bias.detach())

            reg_term_identity = (tau_weight.pow(2).sum() + tau_bias.pow(2).sum())

            reg_term = (self.fisher_dict_cls['weight'](tau_weight) +
                        self.fisher_dict_cls['bias'](tau_bias))

            if do_backward:
                (reg_term * self.req_weight_cls + reg_term_identity * self.simple_reg_weight_cls).backward()

        return reg_term

    def compute_reg_loss(self, do_backward: bool, do_loss_computation: bool):
        return self.tuner.compute_fisher_loss(self.fisher_dict,
                                              do_backward, do_loss_computation)

    def ensemble(self, mode=True):
        self.tuner.ensemble(mode)

    def forward(self, x, train=True, return_all=False,
                return_features=False, use_lora=True, task_weights=None):

        AB = {}

        if use_lora:
            AB = self.tuner.get_lora_matrices(train=train, task_weights=task_weights)

        features = self.vit.forward_features(x, AB)

        if return_features:
            return features

        out = self.vit.forward_head(features)

        if return_all:
            return out, features
        else:
            return out
