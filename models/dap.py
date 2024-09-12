"""
On the reproducibility of DAP
-----------------------------

The original implementation of DAP is available at `https://github.com/naver-ai/dap-cl` and features:
 - a custom backbone, available with the `--load_original_checkpoint` flag
 - majority voting during test time, available with the `--enable_test_time_majority_voting` flag
 - custom splits for many datasets. Specifically: `imagenet-r`, `chestx`, `eurosat-rgb`, `isic`, and `resisc45` use a validation set of 20% from the training set as the test set, while other datasets use their original splits. This differs from the Mammoth implementation, which follows that of other commonly used works (e.g, `eurosat-rgb` uses the split of CoOp).
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from models.dap_utils.dap_model import DAPModel
from utils import binary_to_boolean_type


class DAP(ContinualModel):
    """Generating Instance-level Prompts for Rehearsal-free Continual Learning."""
    NAME = 'dap'
    COMPATIBILITY = ['class-il', 'task-il']
    net: DAPModel

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
        parser.add_argument('--sim_lambda', type=float, default=0.1)  # 2 imgr, 0.5 resisc, 0.1 default
        parser.add_argument("--virtual_bs_n", type=int, default=1, help="virtual batch size iterations")
        parser.add_argument("--enable_test_time_majority_voting", type=int, default=0,
                            help="Enable majority voting for selecting the prompts during test time. NOTE: "
                            "This should be avoided as it is not a fair comparison with other methods.")

        parser.add_argument('--task_emb', type=int, default=16, help='task embedding size')
        parser.add_argument('--num_dap_tokens', type=int, default=10, help='number of dap tokens')

        parser.add_argument('--load_original_checkpoint', type=binary_to_boolean_type, default=0,
                            help='load original checkpoint. This requires the file `imagenet21k_ViT-B_16.npz` to be '
                                 'present in the ./data directory. You can download it following the instructions in '
                                 'https://github.com/naver-ai/dap-cl')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.enable_test_time_majority_voting:
            logging.warning("Majority voting is enabled during test time. The results will not be a fair comparison with other methods.")
        if args.load_original_checkpoint and not os.path.exists('./data/imagenet21k_ViT-B_16.npz'):
            raise FileNotFoundError('`imagenet21k_ViT-B_16.npz` not found in ./data directory. Please follow the instructions in '
                                    'https://github.com/naver-ai/dap-cl to download the file.')
        super(DAP, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.net = DAPModel(backbone=self.net, n_tasks=self.n_tasks, num_classes=self.num_classes, args=args, device=args.device)

        self.opt = self.get_optimizer()

    def get_optimizer(self):
        # check if optimizer is in torch.optim
        _param_groups = [{'params': p, 'lr': self.args.lr} for p in self.get_parameters()]

        opt = torch.optim.Adam(_param_groups, lr=self.args.lr, weight_decay=self.args.optim_wd, betas=(0.9, 0.9), amsgrad=True)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def get_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def begin_task(self, dataset: ContinualDataset) -> None:
        if self.current_task == 1:  # freeze layer after first task
            for k, p in self.net.enc.named_parameters():
                if "dap_downsample" in k:
                    p.requires_grad = False

        # dataset.train_loader.dataset.transform = self.dataset.TEST_TRANSFORM  # transforms.Compose([transforms.ToTensor(), self.dataset.get_normalization_transform()])

        self.net.train()

        self.opt = self.get_optimizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[:, :self.n_seen_classes]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        if self.args.dataset == 'seq-imagenet-r':
            outputs, reduce_sim = self.net(inputs, task_id=self.current_task, is_train=True,
                                           n_past_classes=self.n_past_classes,
                                           n_cur_classes=self.n_cur_classes, is_imgr=True)
        else:
            outputs, reduce_sim = self.net(inputs, task_id=self.current_task, is_train=True)

        outputs[:, :self.n_past_classes] = -np.inf
        outputs[:, self.n_seen_classes:] = -np.inf
        # outputs = outputs[:, self.n_past_classes:self.n_seen_classes]
        loss = self.loss(outputs, labels)

        loss -= self.args.sim_lambda * reduce_sim

        if self.epoch_iteration == 0:
            self.opt.zero_grad()

        (loss / self.args.virtual_bs_n).backward()
        if (self.epoch_iteration > 0 or self.args.virtual_bs_n == 1) and \
                self.epoch_iteration % self.args.virtual_bs_n == 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()
