import torch
from typing import TYPE_CHECKING
from tqdm import tqdm

from functools import partial

from utils.conf import get_device

from models.lora_prototype_utils_v2.vision_transformer_coda_reborn import \
    vit_base_patch16_224_prompt_prototype as vit
from models.lora_prototype_utils_v2.vision_clip import ClipVit as vit_clip

from models.lora_prototype_utils_v2.matching import MatchingEngine
from models.lora_prototype_utils_v2.matching import NullEngine

from models.lora_prototype_utils_v2.loralib.task_lorer import TaskLorer
from models.lora_prototype_utils_v2.loralib.full_lorer import FullLorer
from models.lora_prototype_utils_v2.loralib.ia3_lorer import Ia3Lorer

from models.lora_prototype_utils_v2.utils import IncrementalSemanticClassifier
from models.lora_prototype_utils_v2.fisher import UnbiasedFisherModule
from models.lora_prototype_utils_v2.fisher import AugmentedFisherModule
from models.lora_prototype_utils_v2.fisher import CombinedFisherModule

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset

from torch.nn.functional import softmax


def get_fisher_caller(args):
    if args.augmented_reg == 0:
        fisher_caller = partial(UnbiasedFisherModule, ewc_lambda=args.ewc_lambda)
    elif args.augmented_reg == 1:
        if args.ewc_ensemble_mode == 0:
            fisher_caller = partial(AugmentedFisherModule, ewc_lambda=args.ewc_lambda,
                                    ewc_alpha=args.ewc_alpha, ewc_prior=args.ewc_prior)
        elif args.ewc_ensemble_mode == 1:
            fisher_caller = partial(CombinedFisherModule, ewc_lambda=args.ewc_lambda,
                                    ewc_alpha=args.ewc_alpha, ewc_prior=args.ewc_prior)
        else:
            raise ValueError
    else:
        raise ValueError

    return fisher_caller


class Model(torch.nn.Module):

    def __init__(self, args, seq_dataset: 'ContinualDataset'):

        super().__init__()

        self.compute_fisher = self.__compute_fisher_autograd
        if hasattr(args, "fisher_type"):
            if args.fisher_type == "hooks":
                self.compute_fisher = self.__compute_fisher_hooks

        self.num_classes = seq_dataset.N_CLASSES
        self.num_tasks = seq_dataset.N_TASKS

        self.current_task = 0
        self.class_offsets = [seq_dataset.get_offsets(t) for t in range(self.num_tasks)]
        self.num_classes_per_task = [e - s for (s, e) in self.class_offsets]

        self.device = get_device()

        nc_first_task = self.num_classes_per_task[0]

        if args.adapt_clip == 0:
            self.vit = vit(pretrained=True, num_classes=nc_first_task,
                           args=args).to(self.device)
            self.output_dim = self.vit.embed_dim
            self.embed_dim = self.vit.embed_dim
            self.mlp_ratio = self.vit.mlp_ratio
        else:
            self.vit = vit_clip(args, seq_dataset=seq_dataset,
                                num_classes=nc_first_task).to(self.device)
            self.output_dim = self.vit.output_dim
            self.embed_dim = self.vit.embed_dim
            self.mlp_ratio = 4

        self.adapt_clip = args.adapt_clip
        self.lora_style = args.lora_style

        if self.lora_style in ['standard']:
            self.matching_engine = MatchingEngine(args, self.device, seq_dataset)
        else:
            self.matching_engine = NullEngine()

        self.lorer = self.init_lorer(args, seq_dataset)

        if args.semantic_classifier == 1:
            self.vit.head = IncrementalSemanticClassifier(self.output_dim,
                                                          self.num_classes_per_task[0],
                                                          keys=self.lorer.keys, tau=args.tau)

        param_lored, param_resolution_dict = self.get_params_with_lora()

        self.ewc_fisher_mc_classes = args.ewc_fisher_mc_classes
        self.ewc_cls_lambda = args.ewc_cls_lambda
        self.ewc_cls_identity_lambda = args.ewc_cls_identity_lambda
        self.ewc_cls_prior = args.ewc_cls_prior
        self.ewc_cls_strategy = args.ewc_cls_strategy

        fisher_caller = get_fisher_caller(args)

        self.fisher_dict = torch.nn.ModuleDict({
            param_resolution_dict[n]: fisher_caller(p)
            for n, p in param_lored.items()
        })

        self.fisher_dict_cls = torch.nn.ModuleDict({
            n: UnbiasedFisherModule(p, args.ewc_cls_lambda, args.ewc_cls_prior)
            for n, p in self.vit.head.heads[self.current_task].named_parameters()
        })

        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.iteration = 0

    def set_current_task(self, task_id):
        self.current_task = task_id
        self.lorer.set_current_task(task_id)

        if self.current_task > 0:
            device = self.fisher_dict_cls.weight.unnormalized_fisher.device
            self.fisher_dict_cls.update({
                n: UnbiasedFisherModule(p, self.ewc_cls_lambda, self.ewc_cls_prior).to(device)
                for n, p in self.vit.head.heads[self.current_task].named_parameters()
            })

    def get_output_dim(self):
        return self.output_dim

    def get_embed_dim(self):
        return self.embed_dim

    def init_lorer(self, args, seq_dataset):

        if args.lora_style == 'task':
            return TaskLorer(args, self.device, seq_dataset,
                             embed_dim=self.embed_dim,
                             mlp_ratio=self.mlp_ratio)
        elif args.lora_style == 'full':
            return FullLorer(args, self.device, seq_dataset,
                             embed_dim=self.embed_dim,
                             mlp_ratio=self.mlp_ratio)
        elif args.lora_style == 'ia3':
            return Ia3Lorer(args, self.device, seq_dataset,
                            embed_dim=self.embed_dim,
                            mlp_ratio=self.mlp_ratio,
                            orig_model=self.vit)
        else:
            raise ValueError

    def build_optimizer_args(self, lr_params, lr_classifier,
                             wd_params, wd_classifier):

        lora_vars = self.lorer.get_current_optimizing_parameters()

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

    def get_base_parameters(self):
        return [p for n, p in self.vit.named_parameters() if 'head' not in n]

    def freeze_base_network(self):
        for n, p in self.vit.named_parameters():
            if 'head' not in n:
                p.requires_grad = False

    def unfreeze_base_network(self):
        for n, p in self.vit.named_parameters():
            if 'head' not in n:
                p.requires_grad = True

    def set_requires_grad_to(self, namevars, mode: bool):
        checkset = set()
        for n, p in self.vit.named_parameters():
            if n in namevars:
                p.requires_grad = mode
                checkset.add(n)
        assert checkset == namevars

    def get_rescaled_squared_grads(self, namevars, rescale_factor=1.):
        return {n: rescale_factor * p.grad.pow(2)
                for n, p in self.vit.named_parameters() if n in namevars}

    def get_params_with_lora(self):
        if self.adapt_clip == 0:
            return self.get_params_with_lora_timm()
        return self.get_params_with_lora_openai()

    def get_params_with_lora_timm(self):

        paramname_lored = set()
        param_resolution_dict = dict()

        expansion = {
            'qkv': 'attn.qkv',
            'proj': 'attn.proj',
            'fc1': 'mlp.fc1',
            'fc2': 'mlp.fc2'
        }

        for layer_idx, vars in self.lorer.get_params_with_lora().items():
            for v in vars:
                key_param_name = f'blocks.{layer_idx}.{expansion[v]}.weight'
                paramname_lored.add(key_param_name)
                param_resolution_dict[key_param_name] = f'{v}_{layer_idx}'

        param_lored = {n: p for n, p in self.vit.named_parameters() if n in paramname_lored}

        return param_lored, param_resolution_dict

    def get_params_with_lora_openai(self):
        paramname_lored = set()
        param_resolution_dict = dict()

        expansion = {
            'qkv': 'attn.qkv',
            'proj': 'attn.proj',
            'fc1': 'mlp.c_fc',
            'fc2': 'mlp.c_proj'
        }

        for layer_idx, vars in self.lorer.get_params_with_lora().items():
            for v in vars:
                key_param_name = f'visual.transformer.resblocks.{layer_idx}.{expansion[v]}.weight'
                paramname_lored.add(key_param_name)
                param_resolution_dict[key_param_name] = f'{v}_{layer_idx}'

        param_lored = {n: p for n, p in self.vit.named_parameters() if n in paramname_lored}
        return param_lored, param_resolution_dict

    def train(self, mode=True):
        super().train(False)
        self.lorer.train(False)
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
            if self.ewc_fisher_mc_classes < detached_probs.shape[1]:
                _, class_indices = torch.topk(detached_probs, self.ewc_fisher_mc_classes, dim=1)
                unique, counts = class_indices.unique(return_counts=True)
                unique = unique[torch.argsort(-counts)][:self.ewc_fisher_mc_classes]
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

    def __compute_fisher_autograd(self, param_lored, param_resolution_dict,
                                  param_cls, param_resolution_dict_cls,
                                  dataset, debug_mode: bool = False):

        self.set_requires_grad_to(param_lored.keys(), True)

        fisher = {
            param_resolution_dict[n]: torch.zeros_like(p)
            for (n, p) in param_lored.items()
        }

        fisher_cls = {
            param_resolution_dict_cls[n]: torch.zeros_like(p)
            for (n, p) in param_cls.items()
        }

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in ({**param_lored, **param_cls}).items()],
            lr=0.0
        )

        orig_mode = self.training
        self.eval()

        num_of_examples = 0

        for i, data in tqdm(enumerate(dataset.train_loader),
                            total=len(dataset.train_loader),
                            desc='FISHER computation'):

            if debug_mode and i > 5:
                break

            x, y, _ = data
            x, y = x.to(self.device), y.to(self.device).long()

            num_of_examples += data[0].shape[0]

            for ex, ey in zip(x, y):
                logits = self.forward(ex.unsqueeze(0), train=False, use_lora=False)

                log_probs = self.logsoft(logits)

                num_mc_classes = min(int(logits.shape[1]), self.ewc_fisher_mc_classes)

                with torch.no_grad():
                    probs = softmax(logits, dim=1).squeeze(0)
                    vals, class_indices = torch.topk(probs, num_mc_classes)

                for cnt_class in range(num_mc_classes):
                    fake_optim.zero_grad()
                    y_c = class_indices[cnt_class]
                    log_prob = log_probs[0, y_c]
                    prob = probs[y_c]

                    log_prob.backward(retain_graph=cnt_class < num_mc_classes - 1)

                    rescaled_squared_grads = \
                        self.get_rescaled_squared_grads(param_lored.keys(), prob)

                    for namevar in param_lored:
                        fisher[param_resolution_dict[namevar]] += rescaled_squared_grads[namevar].detach()

                    rescaled_squared_grads = \
                        self.get_rescaled_squared_grads(param_cls.keys(), prob)

                    for namevar in param_cls:
                        fisher_cls[param_resolution_dict_cls[namevar]] += rescaled_squared_grads[namevar].detach()

        fake_optim.zero_grad()

        self.set_requires_grad_to(param_lored.keys(), False)
        self.train(orig_mode)

        del fake_optim

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

    def compute_fisher_cls_generative(self, param_cls, param_resolution_dict_cls, generative_dataloader, debug_mode: bool = False):

        fisher_cls = {
            param_resolution_dict_cls[n]: torch.zeros_like(p)
            for (n, p) in param_cls.items()
        }

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in self.vit.head.named_parameters()],
            lr=0.0
        )

        orig_mode = self.training
        self.eval()

        num_of_examples = 0

        for i, data in tqdm(enumerate(generative_dataloader),
                            total=len(generative_dataloader),
                            desc='FISHER computation'):

            if debug_mode and i > 5:
                break

            x, y = data
            x, y = x.to(self.device), y.to(self.device).long()

            num_of_examples += data[0].shape[0]

            for ex, ey in zip(x, y):
                if debug_mode and i > 5:
                    break

                logits = self.vit.head(ex.unsqueeze(0))

                log_probs = self.logsoft(logits)

                # on all classes
                with torch.no_grad():
                    probs = softmax(logits, dim=1).squeeze(0)

                for cnt_class in range(logits.shape[1]):
                    fake_optim.zero_grad()
                    log_prob = log_probs[0, cnt_class]
                    prob = probs[cnt_class]

                    log_prob.backward(retain_graph=cnt_class < logits.shape[1] - 1)

                    rescaled_squared_grads = \
                        self.get_rescaled_squared_grads(param_cls.keys(), prob)

                    for namevar in param_cls:
                        fisher_cls[param_resolution_dict_cls[namevar]] += rescaled_squared_grads[namevar].detach()

        fake_optim.zero_grad()

        self.train(orig_mode)

        del fake_optim

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

    def update_fisher(self, dataset, generative_dataloader=None, logger_fn=None, debug_mode: bool = False):

        param_lored, param_resolution_dict = self.get_params_with_lora()
        param_cls, param_resolution_dict_cls = self.get_params_of_classifier()

        fisher, fisher_cls, num_of_examples = \
            self.compute_fisher(param_lored, param_resolution_dict,
                                param_cls, param_resolution_dict_cls, dataset, debug_mode)

        num_of_examples_cls = num_of_examples
        if self.ewc_cls_strategy != "bugged":
            for cnt, namevar in enumerate(param_cls.keys()):
                namevar_mapped = param_resolution_dict_cls[namevar]
                fisher_cls[namevar_mapped].zero_()

        if self.ewc_cls_strategy == "generative" and generative_dataloader is not None:
            fisher_cls, num_of_examples_cls = self.compute_fisher_cls_generative_hooks(param_cls, param_resolution_dict_cls, generative_dataloader, debug_mode)
            # fisher_cls, num_of_examples = self.compute_fisher_cls_generative(param_cls, param_resolution_dict_cls, generative_dataloader, debug_mode)

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

            if logger_fn is not None:
                logger_fn({
                    'fisher_norm': (sum_trace / sum_elem).item(),
                    'fisher_norm_cls': (sum_trace_cls / sum_elem_cls).item()
                })

        del fisher
        del fisher_cls

    def update_prevtask_fisher(self, dataset, debug_mode: bool = False):

        param_lored, param_resolution_dict = self.get_params_with_lora()
        param_cls, param_resolution_dict_cls = self.get_params_of_classifier()

        fisher, fisher_cls, num_of_examples = \
            self.compute_fisher(param_lored, param_resolution_dict,
                                param_cls, param_resolution_dict_cls, dataset, debug_mode)

        with torch.no_grad():
            for cnt, namevar in enumerate(param_lored.keys()):
                namevar_mapped = param_resolution_dict[namevar]
                self.fisher_dict[namevar_mapped].update_fisher_prev_task(fisher[namevar_mapped])

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
                (reg_term * self.ewc_cls_lambda + reg_term_identity * self.ewc_cls_identity_lambda).backward()  # TODO: check if term multiplied with cls_prior has grad (should not)

        return reg_term

    def compute_reg_loss(self, do_backward: bool, do_loss_computation: bool):
        return self.lorer.compute_fisher_loss(self.fisher_dict,
                                              do_backward, do_loss_computation)

    def ensemble(self, mode=True):
        self.lorer.ensemble(mode)

    def compute_delta(self, module_param):
        assert "B" in module_param.keys()
        if "A" in module_param.keys():
            B, A = module_param["B"], module_param["A"]
            return (B @ A).sum(0)
        else:
            assert len(module_param.keys()) == 1
            return module_param["B"].squeeze(0)

    @torch.no_grad()
    def add_lora_matrices(self, op: str):
        assert op in ['add', 'sub']

        if self.lora_style in ['standard']:
            raise ValueError

        AB = self.lorer.get_lora_matrices(train=False)

        param_lored, param_resolution_dict = self.get_params_with_lora()
        param_keys_list = list(param_resolution_dict.keys())
        param_values_list = list(param_resolution_dict.values())

        for layer_id, layer_dict in AB.items():
            for module_id, module_param in layer_dict.items():
                delta = self.compute_delta(module_param)
                val_param_dict = f'{module_id}_{layer_id}'
                name_param = param_keys_list[param_values_list.index(val_param_dict)]
                for n1, p1 in self.vit.named_parameters():
                    if n1 == name_param:
                        if op == 'add':
                            p1.add_(delta)
                        else:
                            p1.sub_(delta)

    def merge_lora(self):
        self.add_lora_matrices('add')

    def unmerge_lora(self):
        self.add_lora_matrices('sub')

    def forward(self, x, train=True, return_all=False,
                return_features=False, use_lora=True, task_weights=None):

        AB = {}

        if use_lora:
            AB = self.lorer.get_lora_matrices(train=train, task_weights=task_weights)

        features = self.vit.forward_features(x, AB)

        if return_features:
            return features

        out = self.vit.forward_head(features)

        if return_all:
            return out, features
        else:
            return out
