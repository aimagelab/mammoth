import numpy as np
import torch
from collections import defaultdict

from models.lora_prototype_utils.utils import get_parameter
from models.lora_prototype_utils.loralib.utils import _set_grad_to_zero

class Ia3Lorer(torch.nn.Module):

    def __init__(self, args, device, seq_dataset, embed_dim, mlp_ratio, orig_model):

        super().__init__()

        self.args = args
        self.device = device
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.ensemble_mode = True

        self.current_task = 0

        self.num_tasks = seq_dataset.N_TASKS
        self.scale_by_t = self.args.ewc_scale_by_t == 1

        self.lora_layers = list(range(12))
        self.enable_lora_qkv = True
        self.enable_lora_proj = True
        self.enable_lora_fc = True

        self.lora_config = self.build_lora_config()

        self.register_buffer('ones_buffer', torch.ones((seq_dataset.get_num_classes(),)))
        self.register_buffer('ewc_lambda', torch.ones(1) * args.ewc_lambda)
        self.register_buffer('ewc_alpha', torch.ones(1) * args.ewc_alpha)
        self.register_buffer('ratio', torch.ones(1))

        expansion = {
            'qkv': 'attn.qkv',
            'proj': 'attn.proj',
            'fc1': 'mlp.fc1',
            'fc2': 'mlp.fc2'
        }

        # save theta0 params
        for layer_idx, vars in self.get_params_with_lora().items():
            for v in vars:
                key_param_name = f'blocks.{layer_idx}.{expansion[v]}.weight'
                register_name = f'theta0_L_{v}_{layer_idx}_0'
                self.register_buffer(register_name, orig_model.state_dict()[key_param_name].detach().clone())

        for t in range(self.num_tasks):

            for l in self.lora_layers:
                L_qkv = self.create_AB(self.embed_dim * 3, 1,
                                       enable_op=self.enable_lora_qkv)
                setattr(self, f'L_qkv_{l}_{t}', L_qkv)

                L_proj = self.create_AB(self.embed_dim, 1,
                                        enable_op=self.enable_lora_proj)
                setattr(self, f'L_proj_{l}_{t}', L_proj)

                L_fc1 = self.create_AB(int(mlp_ratio * self.embed_dim), 1,
                                       enable_op=self.enable_lora_fc)
                setattr(self, f'L_fc1_{l}_{t}', L_fc1)

                L_fc2 = self.create_AB(self.embed_dim, 1,
                                       enable_op=self.enable_lora_fc)
                setattr(self, f'L_fc2_{l}_{t}', L_fc2)

                if t == 0:
                    self.register_buffer(f'base_L_qkv_{l}_{t}',
                                         torch.zeros(L_qkv.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'base_L_proj_{l}_{t}',
                                         torch.zeros(L_proj.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'base_L_fc1_{l}_{t}',
                                         torch.zeros(L_fc1.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'base_L_fc2_{l}_{t}',
                                         torch.zeros(L_fc2.shape[:-1] + (int(mlp_ratio * self.embed_dim),)))
                    self.register_buffer(f'basegrad_L_qkv_{l}_{t}',
                                         torch.zeros(L_qkv.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'basegrad_L_proj_{l}_{t}',
                                         torch.zeros(L_proj.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'basegrad_L_fc1_{l}_{t}',
                                         torch.zeros(L_fc1.shape[:-1] + (self.embed_dim,)))
                    self.register_buffer(f'basegrad_L_fc2_{l}_{t}',
                                         torch.zeros(L_fc2.shape[:-1] + (int(mlp_ratio * self.embed_dim),)))

    def create_AB(self, fin, fout,
                  enable_op=True, transpose=False):

        if enable_op is False:
            return None, None

        return get_parameter((1, fin, fout), self.device, 'ones', transpose)

    def set_requires_grad_to_by_task(self, task_id: int, mode: bool):

        for layer_idx in self.lora_layers:
            for namevar, loravals in self.get_lora_config().items():
                if not loravals[0]:
                    continue
                var = getattr(self, f'{namevar}_{layer_idx}_{task_id}')
                var.requires_grad = mode

    def set_current_task(self, current_task: int):
        assert 0 <= current_task < self.num_tasks
        self.current_task = current_task

        if self.current_task == 0:
            ratio = 0.5
        else:
            ratio = 1. / (self.current_task + 1)

        self.ratio.fill_(ratio)

        for task_id in range(self.num_tasks):
            self.set_requires_grad_to_by_task(task_id, task_id == self.current_task)

        if self.current_task == 0:
            return

        with torch.no_grad():
            weights = self.ones_buffer[:self.current_task] / (self.current_task + 1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)

            for layer_idx in self.lora_layers:
                for loravar, loravals in self.get_lora_config().items():
                    if loravals[0] is False:
                        continue
                    mats_past = weights * torch.cat([self._get_matrix(loravar, layer_idx, t).unsqueeze(0)
                                                     for t in range(self.current_task)], dim=0)
                    base_var = getattr(self, f'base_{loravar}_{layer_idx}_0')
                    base_var.copy_(mats_past.sum(0, keepdims=True))

                    basegrad_var = getattr(self, f'basegrad_{loravar}_{layer_idx}_0')
                    basegrad_var.zero_()

                    for t in range(self.current_task):
                        basegrad_var.add_(self._get_matrix(loravar, layer_idx, t).unsqueeze(0))

    def get_current_optimizing_parameters(self):
        params_lora = []
        for layer_idx in self.lora_layers:
            for loravar, loravals in self.get_lora_config().items():
                if loravals[0] is False:
                    continue
                params_lora.append(getattr(self, f'{loravar}_{layer_idx}_{self.current_task}'))
        return params_lora

    def get_current_optimizing_parameters_names(self):
        params_lora = []
        for layer_idx in self.lora_layers:
            for loravar, loravals in self.get_lora_config().items():
                if loravals[0] is False:
                    continue
                params_lora.append(f'{loravar}_{layer_idx}_{self.current_task}')
        return params_lora

    def ensemble(self, mode: bool):
        self.ensemble_mode = mode

    @torch.no_grad()
    def fisher_loss_v1(self, fisher_dict):

        lora_config = self.get_lora_config()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            L_op = f'L_{op}'

            enable_op = lora_config[L_op][0]

            if not enable_op:
                continue

            g = None

            for layer_idx in self.lora_layers:

                fd = fisher_dict[f'{op}_{layer_idx}']
                my_var = getattr(self, f'{L_op}_{layer_idx}_{self.current_task}')
                orig_theta = getattr(self, f'theta0_{L_op}_{layer_idx}_0')

                _set_grad_to_zero(my_var)

                if g is None:
                    g = self._get_matrix(L_op, layer_idx, self.current_task)
                else:
                    g.copy_(self._get_matrix(L_op, layer_idx, self.current_task))

                g.mul_(fd.get_fisher_matrix(scaled=True))

                if self.current_task > 0:
                    g.add_(getattr(self, f'basegrad_{L_op}_{layer_idx}_0').squeeze(0) *
                           fd.get_fisher_matrix_dot(scaled=True),
                           alpha=-1. if fd.dot_is_subbed() else 1.)

                my_var.grad.copy_((g * orig_theta).sum(1, keepdim=True).unsqueeze(0))

                if self.scale_by_t == 1:
                    my_var.grad.mul_(self.ratio)

    def fisher_loss_v2(self, fisher_dict, do_backward: bool, ewc_lambda: float):

        reg_term, dotprod_term = torch.zeros(1), torch.zeros(1)

        lora_config = self.get_lora_config()
        scalar = 0.5 * (1 - self.ratio) * ewc_lambda

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            L_op = f'L_{op}'

            if not lora_config[L_op][0]:
                continue

            for layer_idx in self.lora_layers:

                nets = self._get_matrix(L_op, layer_idx, self.current_task).unsqueeze(0)

                fmod = fisher_dict[f'{op}_{layer_idx}']

                if do_backward:
                    my_var = getattr(self, f'{L_op}_{layer_idx}_{self.current_task}')
                    _set_grad_to_zero(my_var)

                current_reg_loss = scalar * fmod.dist(nets).sum()
                current_dp_loss = 0.

                with torch.no_grad():
                    reg_term += current_reg_loss.detach().cpu()

                if self.current_task > 0:
                    with torch.no_grad():
                        prev_nets = torch.cat([self._get_matrix(L_op, layer_idx, t).unsqueeze(0)
                                               for t in range(self.current_task)], dim=0)
                    current_dp_loss = (self.ratio * (ewc_lambda * fmod.full_dot_prod_no_grad(nets, prev_nets))).sum()
                    with torch.no_grad():
                        dotprod_term += current_dp_loss.detach().cpu()

                if do_backward:
                    (current_reg_loss - current_dp_loss).backward()

                del current_reg_loss
                del current_dp_loss

        return reg_term, dotprod_term

    def compute_fisher_loss(self, fisher_dict,
                            do_backward, do_loss_computation: bool = False):

        reg_term, dotprod_term = torch.zeros(1), torch.zeros(1)
        err_grad = torch.zeros(1)

        if do_backward:
            self.fisher_loss_v1(fisher_dict)

        if not do_loss_computation:
            return reg_term, dotprod_term

        with torch.no_grad():
            reg_term, dotprod_term = self.fisher_loss_v2(fisher_dict, do_backward=False,
                                                         ewc_lambda=self.ones_buffer[0])
        return reg_term, err_grad

    def _get_by_op_layer_and_task(self, op, layer_idx, task_idx):
        L_namevar = f'L_{op}'
        B = self._get_matrix(L_namevar, layer_idx, task_idx)
        return B

    def _get_matrix(self, namevar, layer_idx, task_idx):
        L = getattr(self, f'{namevar}_{layer_idx}_{task_idx}').squeeze(0)
        orig_theta = getattr(self, f'theta0_{namevar}_{layer_idx}_0')
        return orig_theta * (L - 1)

    def get_lora_matrices(self, train=True, task_weights=None):
        return {
            layer_idx: self.get_lora_matrices_by_layer(layer_idx, train, task_weights=task_weights[layer_idx] if task_weights is not None else None)
            for layer_idx in self.lora_layers
        }

    @torch.no_grad()
    def get_weights(self):
        if not self.ensemble_mode:
            return self.ones_buffer[:1].unsqueeze(-1).unsqueeze(-1)
        weights = self.ones_buffer[:(self.current_task + 1)] / (self.current_task + 1)
        return weights.unsqueeze(-1).unsqueeze(-1)

    def get_lora_matrices_by_layer(self, layer_idx, train, task_weights=None):

        params_dict = {
            loravar: [self._gather_matrices(layer_idx, loravar, train, task_weights)
                      if loravals[0] else None] + loravals
            for loravar, loravals in self.get_lora_config().items()
        }

        m = {}

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            L_op = f'L_{op}'

            enable_op = params_dict[L_op][1]

            if not enable_op:
                continue

            L_m = params_dict[L_op][0]

            m[op] = {
                "B": L_m
            }

        return m

    def _gather_matrices(self, layer_idx: int, namevar: str, train: bool,
                         task_weights = None):

        m = self._get_matrix(namevar, layer_idx, self.current_task).unsqueeze(0)

        if self.current_task == 0 or not self.ensemble_mode:
            return m

        if task_weights is not None:
            assert train is False
            weights = self.get_weights()
            weights = weights * task_weights[:, None, None]

            with torch.no_grad():
                mats = torch.stack([self._get_matrix(namevar, layer_idx, t)
                                         for t in range(self.current_task+1)], dim=0)
                return (weights * mats).sum(0, keepdim=True)

        mats_past = getattr(self, f'base_{namevar}_{layer_idx}_0')
        return mats_past + m * (1 / (self.current_task + 1))

    def get_params_with_lora(self):

        params_with_lora = defaultdict(set)

        for layer_idx in self.lora_layers:
            for loravar, loravals in self.get_lora_config().items():
                if loravals[0] is False:
                    continue
                loravars = loravar.split('_')
                assert len(loravars) == 2
                params_with_lora[layer_idx].add(loravars[1])

        return params_with_lora

    def get_lora_config(self):
        return self.lora_config

    def build_lora_config(self):
        return {
            'L_qkv': [self.enable_lora_qkv],
            'L_proj': [self.enable_lora_proj],
            'L_fc1': [self.enable_lora_fc],
            'L_fc2': [self.enable_lora_fc]
        }
