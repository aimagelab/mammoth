# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class SI(ContinualModel):
    """Continual Learning Through Synaptic Intelligence."""
    NAME = 'si'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--c', type=float, required=True,
                            help='surrogate loss weight parameter c')
        parser.add_argument('--xi', type=float, required=True,
                            help='xi parameter for EWC online')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(SI, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

    def penalty(self):  # base form of the penalty term
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.get_params().data - self.checkpoint) ** 2 + self.args.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def get_penalty_grads(self):  # closed form of the gradient of the penalty term
        return self.args.c * 2 * self.big_omega * (self.net.get_params().data - self.checkpoint)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        pre_params = self.net.get_params().detach().data.clone()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        cur_small_omega = self.net.get_grads().data
        if self.big_omega is not None:
            loss_grads = self.net.get_grads()
            self.net.set_grads(loss_grads + self.get_penalty_grads())
        nn.utils.clip_grad.clip_grad_value_(self.get_parameters(), 1)
        self.opt.step()

        cur_small_omega *= (pre_params - self.net.get_params().detach().data.clone())
        self.small_omega += cur_small_omega
        return loss.item()
