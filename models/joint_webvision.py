# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.status import progress_bar
from utils import none_or_float

import timm
from datasets import get_dataset
import os
import sys
import wandb
from datasets.seq_webvision import SequentialWebVision, WebVision, webvision_collate_fn
from utils.conf import base_path_dataset as base_path
import wandb

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1, help='Should use pretrained weights?')
    return parser


class JOINTWebVision(ContinualModel):
    NAME = 'joint_webvision'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):

        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.pretrained = args.pretrained == 1
        backbone = self.get_backbone(args)
        super().__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0
    
    def get_backbone(self, args):
        pretrained = args.pretrained == 1
        return timm.create_model(args.network, pretrained=pretrained, num_classes=self.n_classes)

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS and (len(dataset.test_loaders) != self.args.stop_after if self.args.stop_after is not None else True):
                return

            # reinit network
            self.net = self.get_backbone(self.args)
            self.net.to(self.device)
            self.net.train()
            if self.args.optimizer == 'sgd':
                self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            elif self.args.optimizer == 'adam':
                self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd)
            else:
                raise ValueError(f'Unsupported optimizer: {self.args.optimizer}')

            # prepare dataloader
            train_transform = SequentialWebVision.TRANSFORM
            train_dataset = WebVision(base_path() + 'WebVision', train=True, transform=train_transform, task=-1)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, collate_fn=webvision_collate_fn)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    if self.args.debug_mode and i > 3:
                        break
                    inputs, labels, _ = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.float())
                    if not self.args.nowand:
                        wandb.log({'loss': loss.item()})
                    loss.backward()
                    if self.args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
            
            if self.args.save_checkpoints:
                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                #j = 0 if 'SLURM_JOB_ID' not in os.environ else os.environ['SLURM_JOB_ID']
                t = self.current_task
                j = self.args.conf_jobnum
                print("Saving checkpoint into", f'checkpoints/joint_{self.args.network}_{t}_{j}.pt', file=sys.stderr)
                torch.save(self.net.state_dict(), f'checkpoints/joint_{self.args.network}_{t}_{j}.pt')
                with open(f'checkpoints/joint_{self.args.network}_args_{j}.txt', 'w') as f:
                    print(self.args, file=f)
                
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i + 1) * bs]
                    labels = all_labels[order][i * bs: (i + 1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0
    
    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task-1)
        return self.net(x)[:, :offset_2]
