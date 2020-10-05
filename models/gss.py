# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.gss_buffer import Buffer as Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Gradient based sample selection'
                                        'for online continual learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')
    parser.add_argument('--gss_minibatch_size', type=int, default=None,
                        help='The batch size of the gradient comparison.')
    return parser


class Gss(ContinualModel):
    NAME = 'gss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Gss, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device,
                             self.args.gss_minibatch_size if
                             self.args.gss_minibatch_size is not None
                             else self.args.minibatch_size, self)
        self.alj_nepochs = self.args.batch_num

    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        self.buffer.drop_cache()
        self.buffer.reset_fathom()

        for _ in range(self.alj_nepochs):
            self.opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                tinputs = torch.cat((inputs, buf_inputs))
                tlabels = torch.cat((labels, buf_labels))
            else:
                tinputs = inputs
                tlabels = labels

            outputs = self.net(tinputs)
            loss = self.loss(outputs, tlabels)
            loss.backward()
            self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
