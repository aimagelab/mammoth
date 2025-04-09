from typing import List
import math
from copy import deepcopy

import torch
from torch import nn

from .generative_replay import MixtureOfGaussians


def create_optimizer(optimizer_name, optimizer_arg, momentum=0.9):
    if optimizer_name == 'sgd':
        return torch.optim.SGD(optimizer_arg, momentum=momentum)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(optimizer_arg)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(optimizer_arg)
    else:
        raise ValueError('Optimizer not supported for this method')


def get_parameter(shape, device, type_init: str = 'orto',
                  transpose: bool = False):
    param = torch.zeros(*shape, dtype=torch.float32, device=device)
    if type_init == 'orto':
        torch.nn.init.orthogonal_(param)
    if type_init == 'gaussian':
        torch.nn.init.normal_(param, mean=0.0, std=0.1)
    if type_init == 'kernel':
        torch.nn.init.normal_(param, mean=0.0, std=0.036)
    if type_init == 'attn':
        torch.nn.init.normal_(param, mean=1.0, std=0.03)
    if type_init == 'kaiming':
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    if type_init == 'ones':
        torch.nn.init.ones_(param)
    if transpose:
        param = torch.transpose(param, 1, 2)
    return torch.nn.Parameter(param)


def get_dist(dim: int, n_comp: int = 5, n_iters: int = 500):
    return MixtureOfGaussians(dim, n_components=n_comp,
                              n_iters=n_iters)


def linear_probing_epoch(data_loader, loss_fn, classifier,
                         optim, lr_scheduler, device, debug_mode=False):

    for i, (x, labels) in enumerate(data_loader):
        if debug_mode and i > 10:
            break
        optim.zero_grad()
        x, labels = x.to(device), labels.to(device)
        loss, loss_dict = loss_fn(classifier, x, labels)
        loss.backward()
        optim.step()

    if lr_scheduler:
        lr_scheduler.step()


class IncrementalClassifier(nn.Module):

    def __init__(self, embed_dim, nb_classes, feat_expand=False):

        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand

        heads = [nn.Linear(embed_dim, nb_classes, bias=True)]

        self.heads = nn.ModuleList(heads)
        self.old_state_dict = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def backup(self):
        if self.old_state_dict is not None:
            del self.old_state_dict
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def assign(self, classifier, which_heads: List[int] = None):

        assert len(self.heads) == len(classifier.heads)

        if which_heads is None:
            which_heads = range(len(self.heads))

        for i in which_heads:
            assert 0 <= i < len(self.heads)
            self.heads[i].weight.data.copy_(classifier.heads[i].weight.data)
            self.heads[i].bias.data.copy_(classifier.heads[i].bias.data)

    def get_device(self):
        return next(self.parameters()).device

    def _set_training_state(self, mode):
        for p in self.heads.parameters():
            p.requires_grad = mode

    def enable_training(self):
        self._set_training_state(True)

    def disable_training(self):
        self._set_training_state(False)

    def update(self, nb_classes, freeze_old=True):

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True).to(self.get_device())

        nn.init.trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)

        if freeze_old:
            self.disable_training()

        self.heads.append(_fc)

    def build_optimizer_args(self, lr: float, wd: float = 0):

        params = []

        for ti in range(len(self.heads)):
            current_params = [p for p in self.heads[ti].parameters() if p.requires_grad]
            if len(current_params) > 0:
                params.append({
                    'params': current_params,
                    'lr': lr,
                    'weight_decay': wd
                })

        return params

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        return torch.cat(out, dim=1)


class AlignmentLoss(torch.nn.Module):

    def __init__(self, seq_dataset, device):
        super(AlignmentLoss, self).__init__()
        self.device = device
        self.seq_dataset = seq_dataset

        self.tau_alignment = 0.1
        self.norm_type_alignment = 'all'

        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.current_task = -1
        self.offset_1, self.offset_2 = None, None

    def norm(self, t):
        return torch.norm(t, p=2, dim=-1, keepdim=True) + 1e-7

    def set_current_task(self, current_task):
        self.current_task = current_task
        self.offset_1, self.offset_2 = self.seq_dataset.get_offsets(self.current_task)

    def per_task_norms(self, logits):
        per_task_norm = []
        for _ti in range(self.current_task + 1):
            prev_t_size, cur_t_size = self.seq_dataset.get_offsets(_ti)
            temp_norm = self.norm(logits[:, prev_t_size:cur_t_size])
            per_task_norm.append(temp_norm)
        per_task_norm = torch.cat(per_task_norm, dim=-1)
        norms = per_task_norm.mean(dim=-1, keepdim=True)
        return norms

    def normalize_logits(self, logits):
        if self.norm_type_alignment == 'none':
            return logits
        if self.norm_type_alignment == 'all':
            return logits / (self.tau_alignment * self.norm(logits))
        elif self.norm_type_alignment == 'pertask':
            return logits / (self.tau_alignment * self.per_task_norms(logits))
        else:
            raise ValueError

    def forward(self, classifier, features, labels):
        loss_dict = {}
        ce_logits = classifier(features)
        ce_logits = self.normalize_logits(ce_logits)
        loss = self.cross_entropy(ce_logits, labels)
        loss_dict['stream_loss'] = loss.item()

        return loss, loss_dict
