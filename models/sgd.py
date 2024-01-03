"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Sgd(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
