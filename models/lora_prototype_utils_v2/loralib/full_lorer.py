import torch
from collections import defaultdict

from models.lora_prototype_utils_v2.utils import get_parameter
from models.lora_prototype_utils_v2.loralib.utils import _set_grad_to_zero


class FullLorer(torch.nn.Module):

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
                BA_qkv = self.create_BA(self.embed_dim, self.embed_dim * 3)
                BA_proj = self.create_BA(self.embed_dim, self.embed_dim)
                BA_fc1 = self.create_BA(self.embed_dim, int(mlp_ratio * self.embed_dim))
                BA_fc2 = self.create_BA(int(mlp_ratio * self.embed_dim), self.embed_dim)

                setattr(self, f'B_qkv_{l}_{t}', BA_qkv)
                setattr(self, f'B_proj_{l}_{t}', BA_proj)
                setattr(self, f'B_fc1_{l}_{t}', BA_fc1)
                setattr(self, f'B_fc2_{l}_{t}', BA_fc2)

                if t == 0:
                    self.register_buffer(f'base_B_qkv_{l}_{t}', torch.zeros(BA_qkv.shape))
                    self.register_buffer(f'base_B_proj_{l}_{t}', torch.zeros(BA_proj.shape))
                    self.register_buffer(f'base_B_fc1_{l}_{t}', torch.zeros(BA_fc1.shape))
                    self.register_buffer(f'base_B_fc2_{l}_{t}', torch.zeros(BA_fc2.shape))
                    self.register_buffer(f'basegrad_B_qkv_{l}_{t}', torch.zeros(BA_qkv.shape))
                    self.register_buffer(f'basegrad_B_proj_{l}_{t}', torch.zeros(BA_proj.shape))
                    self.register_buffer(f'basegrad_B_fc1_{l}_{t}', torch.zeros(BA_fc1.shape))
                    self.register_buffer(f'basegrad_B_fc2_{l}_{t}', torch.zeros(BA_fc2.shape))

    def create_BA(self, fin, fout, transpose=False):
        return get_parameter((1, fout, fin), self.device, 'zeros', transpose)

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
            self.set_requires_grad_to_by_task(task_id,
                                              task_id == self.current_task)

        if self.current_task == 0:
            return

        with torch.no_grad():
            weights = self.ones_buffer[:self.current_task] / (self.current_task + 1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)

            for layer_idx in self.lora_layers:
                for loravar, loravals in self.get_lora_config().items():
                    if loravals[0] is False:
                        continue
                    mats_past = weights * torch.cat([self._get_matrix(loravar, layer_idx, t)
                                                     for t in range(self.current_task)], dim=0)
                    base_var = getattr(self, f'base_{loravar}_{layer_idx}_0')
                    base_var.copy_(mats_past.sum(0, keepdims=True))

                    basegrad_var = getattr(self, f'basegrad_{loravar}_{layer_idx}_0')
                    basegrad_var.zero_()

                    for t in range(self.current_task):
                        basegrad_var.add_(self._get_matrix(loravar, layer_idx, t))

    def get_current_optimizing_parameters(self):
        params_lora = []
        for layer_idx in self.lora_layers:
            for loravar, loravals in self.get_lora_config().items():
                if loravals[0] is False:
                    continue
                params_lora.append(getattr(self, f'{loravar}_{layer_idx}_{self.current_task}'))
        return params_lora

    def ensemble(self, mode: bool):
        self.ensemble_mode = mode

    @torch.no_grad()
    def fisher_loss_v1(self, fisher_dict):

        lora_config = self.get_lora_config()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            B_op = f'B_{op}'

            enable_op = lora_config[B_op][0]

            if not enable_op:
                continue

            g = None

            for layer_idx in self.lora_layers:

                fd = fisher_dict[f'{op}_{layer_idx}']

                my_var = self._get_matrix(B_op, layer_idx, self.current_task)

                _set_grad_to_zero(my_var)

                if g is None:
                    g = my_var
                else:
                    g.copy_(my_var)

                g.mul_(fd.get_fisher_matrix(scaled=True).unsqueeze(0))

                if self.current_task > 0:
                    g.add_(getattr(self, f'basegrad_{B_op}_{layer_idx}_0') *
                           fd.get_fisher_matrix_dot(scaled=True).unsqueeze(0),
                           alpha=-1. if fd.dot_is_subbed() else 1.)

                my_var.grad.copy_(g)

                if self.scale_by_t == 1:
                    my_var.grad.mul_(self.ratio)

    def fisher_loss_v2(self, fisher_dict, do_backward: bool, ewc_lambda: float):

        reg_term, dotprod_term = torch.zeros(1), torch.zeros(1)

        lora_config = self.get_lora_config()

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            B_op = f'B_{op}'

            if not lora_config[B_op][0]:
                continue

            for layer_idx in self.lora_layers:

                nets = self._get_matrix(B_op, layer_idx, self.current_task)

                fmod = fisher_dict[f'{op}_{layer_idx}']

                current_reg_loss = 0.5 * ((1 - self.ratio) * (ewc_lambda * fmod.dist(nets))).sum()
                current_dp_loss = 0.

                with torch.no_grad():
                    reg_term += current_reg_loss.detach().cpu()

                if self.current_task > 0:
                    with torch.no_grad():
                        prev_nets = torch.cat([self._get_matrix(B_op, layer_idx, t)
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

        if do_backward:
            self.fisher_loss_v1(fisher_dict)

        if not do_loss_computation:
            return torch.zeros(1), torch.zeros(1)

        with torch.no_grad():
            reg_term, dotprod_term = self.fisher_loss_v2(fisher_dict, do_backward=False,
                                                         ewc_lambda=self.ones_buffer[0])

        return reg_term, dotprod_term

    def _get_matrix(self, namevar, layer_idx, task_idx):
        return getattr(self, f'{namevar}_{layer_idx}_{task_idx}')

    def get_lora_matrices(self, train=True, task_weights=None):
        return {
            layer_idx: self.get_lora_matrices_by_layer(layer_idx, train)
            for layer_idx in self.lora_layers
        }

    def get_lora_matrices_by_layer(self, layer_idx, train):

        params_dict = {
            loravar: [self._gather_matrices(layer_idx, loravar, train)
                      if loravals[0] else None] + loravals
            for loravar, loravals in self.get_lora_config().items()
        }

        m = {}

        for op in ['qkv', 'proj', 'fc1', 'fc2']:

            B_op = f'B_{op}'

            if not params_dict[B_op][1]:
                continue

            m[op] = {
                "B": params_dict[B_op][0]
            }

        return m

    def _gather_matrices(self, layer_idx: int, namevar: str, train: bool):

        m = self._get_matrix(namevar, layer_idx, self.current_task)

        if self.current_task == 0 or not self.ensemble_mode:
            return m

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
            'B_qkv': [self.enable_lora_qkv],
            'B_proj': [self.enable_lora_proj],
            'B_fc1': [self.enable_lora_fc],
            'B_fc2': [self.enable_lora_fc]
        }
