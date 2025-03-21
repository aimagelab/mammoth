import numpy as np
import torch
from collections import defaultdict

from models.lora_prototype_utils.utils import get_parameter
from models.lora_prototype_utils.loralib.utils import _set_grad_to_zero


class TaskLorer(torch.nn.Module):

    def __init__(self, args, device, seq_dataset, embed_dim, mlp_ratio):

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

        self.register_buffer('ones_buffer', torch.ones((seq_dataset.N_CLASSES,)))
        self.register_buffer('ewc_lambda', torch.ones(1) * args.ewc_lambda)
        self.register_buffer('ratio', torch.ones(1))

        for t in range(self.num_tasks):

            for l in self.lora_layers:

                AB_qkv = self.create_AB(self.embed_dim, self.embed_dim * 3,
                                        r1=self.args.lora_r * 3, r2=self.args.lora_r * 3,
                                        enable_op=self.enable_lora_qkv)

                setattr(self, f'A_qkv_{l}_{t}', AB_qkv[0])
                setattr(self, f'B_qkv_{l}_{t}', AB_qkv[1])

                AB_proj = self.create_AB(self.embed_dim, self.embed_dim,
                                         enable_op=self.enable_lora_proj)

                setattr(self, f'A_proj_{l}_{t}', AB_proj[0])
                setattr(self, f'B_proj_{l}_{t}', AB_proj[1])

                AB_fc1 = self.create_AB(self.embed_dim, int(mlp_ratio * self.embed_dim),
                                        enable_op=self.enable_lora_fc)

                setattr(self, f'A_fc1_{l}_{t}', AB_fc1[0])
                setattr(self, f'B_fc1_{l}_{t}', AB_fc1[1])

                AB_fc2 = self.create_AB(int(mlp_ratio * self.embed_dim), self.embed_dim,
                                        enable_op=self.enable_lora_fc)

                setattr(self, f'A_fc2_{l}_{t}', AB_fc2[0])
                setattr(self, f'B_fc2_{l}_{t}', AB_fc2[1])

                if t == 0:
                    self.register_buffer(f'basegrad_qkv_{l}_{t}',
                                         torch.zeros((1, self.embed_dim * 3, self.embed_dim)))
                    self.register_buffer(f'basegrad_proj_{l}_{t}',
                                         torch.zeros((1, self.embed_dim, self.embed_dim)))
                    self.register_buffer(f'basegrad_fc1_{l}_{t}',
                                         torch.zeros((1, int(mlp_ratio * self.embed_dim), self.embed_dim)))
                    self.register_buffer(f'basegrad_fc2_{l}_{t}',
                                         torch.zeros((1, self.embed_dim, int(mlp_ratio * self.embed_dim))))

    def create_AB(self, fin, fout, r1=None, r2=None,
                  enable_op=True, transpose=False):

        if enable_op is False:
            return None, None

        r1 = self.args.lora_r if r1 is None else r1
        r2 = self.args.lora_r if r2 is None else r2

        config = ('kaiming', 'zeros')

        return get_parameter((1, r1, fin), self.device, config[0], transpose), \
            get_parameter((1, fout, r2), self.device, config[1], transpose)

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

        with torch.no_grad():

            for layer_idx in self.lora_layers:
                for loravar, loravals in self.get_lora_config().items():
                    if loravals[0] is False:
                        continue

                    loravars = loravar.split('_')
                    assert len(loravars) == 2

                    if loravars[0] == 'A':
                        basegrad_var = getattr(self, f'basegrad_{loravars[1]}_{layer_idx}_0')
                        basegrad_var.zero_()

                        for t in range(self.current_task):
                            B_past, A_past = \
                                self._get_by_op_layer_and_task(loravars[1], layer_idx,
                                                               t, return_splitted=True)
                            basegrad_var.add_(B_past @ A_past)

        for task_id in range(self.num_tasks):
            self.set_requires_grad_to_by_task(task_id, task_id == self.current_task)

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

            A_op, B_op = f'A_{op}', f'B_{op}'

            enable_op = lora_config[A_op][0] and lora_config[B_op][0]

            if not enable_op:
                continue

            g = None

            for layer_idx in self.lora_layers:

                fd = fisher_dict[f'{op}_{layer_idx}']

                var_A = getattr(self, f'A_{op}_{layer_idx}_{self.current_task}')
                var_B = getattr(self, f'B_{op}_{layer_idx}_{self.current_task}')

                _set_grad_to_zero(var_A)
                _set_grad_to_zero(var_B)

                B, A = self._get_by_op_layer_and_task(op, layer_idx,
                                                      self.current_task, return_splitted=True)
                if g is None:
                    g = B @ A
                else:
                    g.copy_(B @ A)

                g.mul_(fd.get_fisher_matrix(scaled=True).unsqueeze(0))

                if self.current_task > 0:
                    g.add_(getattr(self, f'basegrad_{op}_{layer_idx}_0') *
                           fd.get_fisher_matrix_dot(scaled=True).unsqueeze(0),
                           alpha=-1. if fd.dot_is_subbed() else 1.)

                var_A.grad.copy_(B.transpose(1, 2) @ g)
                var_B.grad.copy_(g @ A.transpose(1, 2))

                if self.scale_by_t == 1:
                    var_A.grad.mul_(self.ratio)
                    var_B.grad.mul_(self.ratio)

    def fisher_loss_v2(self, fisher_dict, do_backward: bool, ewc_lambda: float):

        reg_term, dotprod_term = torch.zeros(1), torch.zeros(1)

        lora_config = self.get_lora_config()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            A_op, B_op = f'A_{op}', f'B_{op}'

            enable_op = lora_config[A_op][0] and lora_config[B_op][0]

            if not enable_op:
                continue

            for layer_idx in self.lora_layers:

                nets = self._get_by_op_layer_and_task(op, layer_idx, self.current_task)

                fmod = fisher_dict[f'{op}_{layer_idx}']

                current_reg_loss = 0.5 * ((1 - self.ratio) * (ewc_lambda * fmod.dist(nets))).sum()
                current_dp_loss = 0

                with torch.no_grad():
                    reg_term += current_reg_loss.detach().cpu()

                if self.current_task > 0:
                    with torch.no_grad():
                        prev_nets = torch.cat([self._get_by_op_layer_and_task(op, layer_idx, t)
                                               for t in range(self.current_task)], dim=0)
                    current_dp_loss = (self.ratio *
                                       (ewc_lambda * fmod.full_dot_prod_no_grad(nets, prev_nets))).sum()

                    with torch.no_grad():
                        dotprod_term += current_dp_loss.detach().cpu()

                if do_backward:
                    (current_reg_loss - current_dp_loss).backward()

                del current_dp_loss
                del current_reg_loss

        return reg_term, dotprod_term

    def compute_identity_fisher(self):
        reg_term = torch.zeros(1)
        lora_config = self.get_lora_config()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            A_op, B_op = f'A_{op}', f'B_{op}'

            enable_op = lora_config[A_op][0] and lora_config[B_op][0]

            if not enable_op:
                continue

            for layer_idx in self.lora_layers:
                var_A = getattr(self, f'A_{op}_{layer_idx}_{self.current_task}')
                var_B = getattr(self, f'B_{op}_{layer_idx}_{self.current_task}')

                _set_grad_to_zero(var_A)
                _set_grad_to_zero(var_B)

                nets = self._get_by_op_layer_and_task(op, layer_idx, self.current_task)
                base_loss = nets.pow(2).sum()
                if self.args.ewc_alpha != 0:
                    loss = self.args.ewc_alpha * base_loss
                    loss.backward()
                reg_term += base_loss.detach().cpu()

        return reg_term

    def compute_fisher_loss(self, fisher_dict, do_backward,
                            do_loss_computation: bool = False):
        if self.args.fisher_type == 'identity':
            reg_term = self.compute_identity_fisher()
            return reg_term, torch.zeros(1)

        if do_backward:
            self.fisher_loss_v1(fisher_dict)

        if not do_loss_computation:
            return torch.zeros(1), torch.zeros(1)

        with torch.no_grad():
            reg_term, dotprod_term = self.fisher_loss_v2(fisher_dict, do_backward=False,
                                                         ewc_lambda=self.ones_buffer[0])
        return reg_term, dotprod_term

    def _get_by_op_layer_and_task(self, op, layer_idx, task_idx,
                                  return_splitted: bool = False):
        A_namevar, B_namevar = f'A_{op}', f'B_{op}'
        A = self._get_matrix(A_namevar, layer_idx, task_idx)
        B = self._get_matrix(B_namevar, layer_idx, task_idx)
        if return_splitted:
            return B, A
        return B @ A

    def _get_matrix(self, namevar, layer_idx, task_idx):
        return getattr(self, f'{namevar}_{layer_idx}_{task_idx}')

    def get_lora_matrices(self, train=True, task_weights=None, retain_grad=False):
        return {
            layer_idx: self.get_lora_matrices_by_layer(layer_idx, train, task_weights=task_weights[layer_idx] if task_weights is not None else None, retain_grad=retain_grad)
            for layer_idx in self.lora_layers
        }

    @torch.no_grad()
    def get_weights(self):
        if not self.ensemble_mode:
            return self.ones_buffer[:1].unsqueeze(-1).unsqueeze(-1)
        weights = self.ones_buffer[:(self.current_task + 1)] / (self.current_task + 1)
        return weights.unsqueeze(-1).unsqueeze(-1)

    def get_lora_matrices_by_layer(self, layer_idx, train, task_weights=None, retain_grad=False):

        params_dict = {
            loravar: [self._gather_matrices(layer_idx, loravar, train)
                      if loravals[0] else None] + loravals
            for loravar, loravals in self.get_lora_config().items()
        }

        m = {}

        weights = self.get_weights()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            A_op, B_op = f'A_{op}', f'B_{op}'

            enable_op = params_dict[A_op][1] and params_dict[B_op][1]

            if not enable_op:
                continue

            A_m = params_dict[A_op][0]
            B_m = params_dict[B_op][0]

            if task_weights is not None:
                w = weights * task_weights[:, None, None]
            else:
                w = weights
            A: torch.Tensor = A_m * w
            B: torch.Tensor = B_m
            if retain_grad:
                A.retain_grad()
                B.retain_grad()
            m[op] = {
                "B": B,
                "A": A
            }

        return m

    def _gather_matrices(self, layer_idx: int, namevar: str, train: bool):

        m = self._get_matrix(namevar, layer_idx, self.current_task)

        if self.current_task == 0 or not self.ensemble_mode:
            return m

        with torch.no_grad():
            mats_past = torch.cat([self._get_matrix(namevar, layer_idx, t)
                                   for t in range(self.current_task)], dim=0)

        return torch.cat([mats_past.detach(), m], dim=0)

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
            'A_qkv': [self.enable_lora_qkv],
            'B_qkv': [self.enable_lora_qkv],
            'A_proj': [self.enable_lora_proj],
            'B_proj': [self.enable_lora_proj],
            'A_fc1': [self.enable_lora_fc],
            'B_fc1': [self.enable_lora_fc],
            'A_fc2': [self.enable_lora_fc],
            'B_fc2': [self.enable_lora_fc]
        }
