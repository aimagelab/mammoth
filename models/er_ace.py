# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class CustomLinear(torch.nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super(CustomLinear, self).__init__()
        self.L = torch.nn.Linear(indim, outdim, bias=False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x: torch.Tensor):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        cos_dist = torch.mm(x_normalized, self.L.weight.div(L_norm + 0.00001).transpose(0, 1))

        scores = self.scale_factor * (cos_dist)

        return scores


class ErACE(ContinualModel):
    """Continual learning via Experience Replay with asymmetric cross-entropy."""
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--task_free', type=binary_to_boolean_type, default=False, help='Enable task-free training (replay starts from second task)?.')
        parser.add_argument('--use_custom_classifier', type=binary_to_boolean_type, default=True, help='Use the custom classifier used in the original work.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.use_custom_classifier:
            assert hasattr(backbone, 'classifier'), 'The backbone must have a classifier layer.'
            backbone.classifier = CustomLinear(backbone.classifier.in_features, backbone.classifier.out_features)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.seen_so_far = torch.tensor([]).long().to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        # if self.seen_so_far.max() < (self.num_classes - 1):
        mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0 or self.args.task_free:
            logits = logits.masked_fill(mask == 0, -1e9)  # torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if len(self.buffer) > 0:
            if self.args.task_free or self.current_task > 0:
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
