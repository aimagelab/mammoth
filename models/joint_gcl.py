# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from backbone import get_backbone
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.status import progress_bar


class JointGCL(ContinualModel):
    """Joint training: a strong, simple baseline."""
    NAME = 'joint_gcl'
    COMPATIBILITY = ['general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(n_epochs=1)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(JointGCL, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.old_data = []
        self.old_labels = []

    def end_task(self, dataset):
        # reinit network
        self.net = get_backbone(self.args)
        self.net.to(self.device)
        self.net.train()
        self.opt = self.get_optimizer()

        # gather data
        all_data = torch.cat(self.old_data)
        all_labels = torch.cat(self.old_labels)

        # train (single epochs because GCL)
        rp = torch.randperm(len(all_data))
        for i in range(math.ceil(len(all_data) / self.args.batch_size)):
            inputs = all_data[rp][i * self.args.batch_size:(i + 1) * self.args.batch_size]
            labels = all_labels[rp][i * self.args.batch_size:(i + 1) * self.args.batch_size]
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            progress_bar(i, math.ceil(len(all_data) / self.args.batch_size), 0, 'J', loss.item())

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.old_data.append(inputs.data)
        self.old_labels.append(labels.data)
        return 0
