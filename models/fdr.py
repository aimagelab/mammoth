# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Fdr(ContinualModel):
    NAME = 'fdr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Fdr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.i = 0
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def end_task(self, dataset):
        self.current_task += 1
        examples_per_task = self.args.buffer_size // self.current_task

        if self.current_task > 1:
            buf_x, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs = inputs.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(inputs)
                if examples_per_task - counter < 0:
                    break
                self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                     logits=outputs.data[:(examples_per_task - counter)],
                                     task_labels=(torch.ones(self.args.batch_size) *
                                                  (self.current_task - 1))[:(examples_per_task - counter)])
                counter += self.args.batch_size

    def observe(self, inputs, labels, not_aug_inputs):
        self.i += 1

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        if not self.buffer.is_empty():
            self.opt.zero_grad()
            buf_inputs, buf_logits, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss = torch.norm(self.soft(buf_outputs) - self.soft(buf_logits), 2, 1).mean()
            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()

        return loss.item()
