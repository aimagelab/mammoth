"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform, dataset)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def apply_decay(decay, lr, optimizer, num_iter):
    if decay != 1:
        learn_rate = lr * (decay ** num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_rate


class ErAceTricks(ContinualModel):
    """Experience Replay with tricks from `Rethinking Experience Replay: a Bag of Tricks for Continual Learning`."""
    NAME = 'er_ace_tricks'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.
        """
        add_rehearsal_args(parser)

        parser.add_argument('--bic_epochs', type=int, default=50, help='bias injector.')
        parser.add_argument('--elrd', type=float, default=0.99999925)

        parser.add_argument('--sample_selection_strategy', default='labrs', type=str, choices=['reservoir', 'lars', 'labrs'],
                            help='Sample selection strategy to use: `reservoir`, `lars` (Loss-Aware Reservoir Sampling), `labrs` (Loss-Aware Balanced Reservoir Sampling)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErAceTricks, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, device=self.device, sample_selection_strategy=self.args.sample_selection_strategy)

        # BIC
        self.bic_params = torch.zeros(2, device=self.device, requires_grad=True)
        self.bic_opt = torch.optim.SGD([self.bic_params], lr=0.5)

        # ACE
        self.seen_so_far = torch.tensor([]).long().to(self.device)

    def end_task(self, dataset):
        self.net.eval()
        for l in range(self.args.bic_epochs):
            data = self.buffer.get_data(self.args.buffer_size, transform=dataset.get_normalization_transform())
            while data[0].shape[0] > 0:
                inputs, labels = data[0][:self.args.batch_size], data[1][:self.args.batch_size]
                data = (data[0][self.args.batch_size:], data[1][self.args.batch_size:])

                self.bic_opt.zero_grad()
                with torch.no_grad():
                    out = self.net(inputs)

                out[:, self.n_past_classes:self.n_seen_classes] *= self.bic_params[1].repeat_interleave(self.n_classes_current_task)
                out[:, self.n_past_classes:self.n_seen_classes] += self.bic_params[0].repeat_interleave(self.n_classes_current_task)

                loss_bic = self.loss(out, labels)
                loss_bic.backward()
                self.bic_opt.step()

        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        apply_decay(self.args.elrd, self.args.lr, self.opt, self.buffer.num_seen_examples)

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        self.opt.zero_grad()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        stream_loss_scores = self.loss(logits, labels, reduction='none')
        loss = stream_loss_scores.mean()

        loss_re = torch.tensor(0.)

        if not self.buffer.is_empty():
            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device, return_index=True)

            buf_loss_scores = self.loss(self.net(buf_inputs), buf_labels)
            loss_re = buf_loss_scores.mean()
            loss += loss_re

        loss.backward()
        self.opt.step()

        if not self.buffer.is_empty():
            self.buffer.sample_selection_fn.update(buf_indexes, -buf_loss_scores.detach())
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             sample_selection_scores=-stream_loss_scores.detach())

        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = super(ErAceTricks, self).forward(x)
        if ret.shape[0] > 0:
            ret[:, self.n_past_classes:self.n_seen_classes] *= self.bic_params[1].repeat_interleave(self.n_classes_current_task)
            ret[:, self.n_past_classes:self.n_seen_classes] += self.bic_params[0].repeat_interleave(self.n_classes_current_task)

        return ret
