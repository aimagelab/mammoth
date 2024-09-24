# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class Der(ContinualModel):
    """Continual learning via Dark Experience Replay."""
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Der, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_mse

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
