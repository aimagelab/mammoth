# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.args import *
from models.utils.continual_model import ContinualModel


class LwFMC(ContinualModel):
    """Learning without Forgetting - Multi-Class."""
    NAME = 'lwf_mc'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--wd_reg', type=float, default=0.0,
                            help='L2 regularization applied to the parameters.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LwFMC, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, logits)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index

        Returns:
            the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, :ac]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        if self.args.wd_reg:
            loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        return loss

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
