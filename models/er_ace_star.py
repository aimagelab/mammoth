# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.star_utils.star_perturber import Perturber, add_perturb_args


class ErACESTAR(ContinualModel):
    NAME = 'er_ace_star'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # add arguments for STAR
        add_perturb_args(parser)
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.pert = Perturber(self)
        self.seen_so_far = torch.tensor([]).long().to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            # STAR here
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            self.pert(buf_inputs, buf_labels)

        # normal er_ace resumes

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.current_task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

            loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()
