# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    """Continual learning via A-GEM."""
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AGem, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()

        return loss.item()
