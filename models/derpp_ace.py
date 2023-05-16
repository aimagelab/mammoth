# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
import timm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--immediate_replay', type=int, default=0,
                        help='Penalty weight.')
    parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1, help='Should use pretrained weights?')
    return parser


class DerppAce(ContinualModel):
    NAME = 'derpp_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        backbone = timm.create_model(args.network, pretrained=args.pretrained==1, num_classes=self.n_classes)
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_y_so_far = torch.zeros(self.n_classes).bool().to(self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.current_task = 0
        self.buf_transform = self.dataset.TRANSFORM
        self.opt = self.get_optimizer()
    
    def begin_task(self, dataset):
        if self.current_task == 0:
            self.opt = self.get_optimizer()
        self.old_epoch = 0

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.current_task += 1
    
    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task-1)
        return self.net(x)[:, :offset_2]
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        logits = self.net(inputs)
        output_mask = self.seen_y_so_far.unsqueeze(0).expand_as(logits).detach().clone()
        filtered_logits = logits[:, :offset_2]
        idx = labels.sum(0).nonzero().squeeze(1)
        filtered_output = filtered_logits[:, idx]
        filtered_target = labels[:, idx]
        loss = self.loss(filtered_output, filtered_target.float())
        self.seen_y_so_far[:offset_2] |= labels[:, :offset_2].any(dim=0).data

        loss_der = torch.tensor(0.0).to(self.device)
        loss_re = torch.tensor(0.0).to(self.device)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, buf_logits_mask = self.buffer.get_data(
                self.args.batch_size, transform=self.buf_transform) 
            buf_outputs = self.net(buf_inputs)
            # ignore unseen classes in targets
            buf_logits[~buf_logits_mask] = 0.0
            buf_outputs[~buf_logits_mask] = 0.0
            loss_der = F.mse_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * loss_der

            buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                self.args.batch_size, transform=self.buf_transform)
            
            loss_re = self.loss(self.net(buf_inputs), buf_labels.float())
            loss += self.args.beta * loss_re

        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_grad'])
        self.opt.step()

        if output_mask.sum() > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=logits.data,
                                logits_mask=output_mask.data)

        return loss.item()
