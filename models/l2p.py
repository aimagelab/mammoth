# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import models.l2p_utils.vit_prompt  # required to register the models
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from datasets import get_dataset
import numpy as np
from models.l2p_utils.l2p_model import L2PModel
from utils import none_or_float


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning to Prompt (L2P)')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Prompt parameters
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--pool_size_l2p', default=10, type=int,)
    parser.add_argument('--length', default=5, type=int, )
    parser.add_argument('--top_k', default=5, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str,)
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    #parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')
    parser.add_argument('--clip_grad', type=none_or_float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    return parser


class L2P(ContinualModel):
    NAME = 'l2p'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        assert self.dataset.SIZE[0] >= 224, 'L2P only supports 224x224 images or greater'
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS

        backbone = L2PModel(args, self.n_classes)

        super().__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.class_mask = torch.arange(self.n_classes, dtype=int) \
            .reshape(self.dataset.N_TASKS, self.n_classes // self.dataset.N_TASKS).tolist()

    def begin_task(self, dataset):
        self.net.original_model.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        outputs = self.net(inputs, return_outputs=True)
        logits = outputs['logits']

        # here is the trick to mask out classes of non-current tasks
        if self.class_mask is not None:
            mask = self.class_mask[self.current_task]
            not_mask = np.setdiff1d(np.arange(self.n_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        offset_1, offset_2 = self._compute_offsets(self.current_task)
        logits = logits[:, offset_1:offset_2]

        loss = self.loss(logits, labels)
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            loss = loss - self.args.pull_constraint_coeff * outputs['reduce_sim']

        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.model.parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        self.current_task += 1

    def forward(self, x):
        if self.current_task > 0:
            offset_1, offset_2 = self._compute_offsets(self.current_task - 1)
        else:
            offset_2 = self.N_CLASSES
        return self.net(x)[:, :offset_2]
