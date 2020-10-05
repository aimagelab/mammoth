# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Meta-Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # remove batch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'batch_size':
            del parser._actions[i]
            break

    parser.add_argument('--beta', type=float, required=True,
                        help='Within-batch update beta parameter.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Across-batch update gamma parameter.')
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')

    return parser


class Mer(ContinualModel):
    NAME = 'mer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Mer, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def draw_batches(self, inp, lab):
        batches = []
        for i in range(self.args.batch_num):
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                inputs = torch.cat((buf_inputs, inp.unsqueeze(0)))
                labels = torch.cat((buf_labels, torch.tensor([lab]).to(self.device)))
                batches.append((inputs, labels))
            else:
                batches.append((inp.unsqueeze(0), torch.tensor([lab]).unsqueeze(0).to(self.device)))
        return batches

    def observe(self, inputs, labels, not_aug_inputs):

        batches = self.draw_batches(inputs, labels)
        theta_A0 = self.net.get_params().data.clone()

        for i in range(self.args.batch_num):
            theta_Wi0 = self.net.get_params().data.clone()

            batch_inputs, batch_labels = batches[i]

            # within-batch step
            self.opt.zero_grad()
            outputs = self.net(batch_inputs)
            loss = self.loss(outputs, batch_labels.squeeze(-1))
            loss.backward()
            self.opt.step()

            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
            self.net.set_params(new_params)

        self.buffer.add_data(examples=not_aug_inputs.unsqueeze(0), labels=labels)

        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (self.net.get_params() - theta_A0)
        self.net.set_params(new_new_params)

        return loss.item()
