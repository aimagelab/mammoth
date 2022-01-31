# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

def dsimplex(num_classes=10):
    def simplex_coordinates2(m):
        # add the credit
        import numpy as np

        x = np.zeros([m, m + 1])
        for j in range(0, m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)

        for i in range(0, m):
            x[i, m] = a

        #  Adjust coordinates so the centroid is at zero.
        c = np.zeros(m)
        for i in range(0, m):
            s = 0.0
            for j in range(0, m + 1):
                s = s + x[i, j]
            c[i] = s / float(m + 1)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] - c[i]

        #  Scale so each column has norm 1. UNIT NORMALIZED
        s = 0.0
        for i in range(0, m):
            s = s + x[i, 0] ** 2
        s = np.sqrt(s)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] / s

        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class RPC(ContinualModel):
    NAME = 'rpc'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(RPC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.tasks = get_dataset(args).N_TASKS
        self.task=0
        self.rpchead = torch.from_numpy(dsimplex(self.cpt * self.tasks)).float().to(self.device)

    def forward(self, x):
        x = self.net(x)[:, :-1]
        x = x @ self.rpchead
        return x

    def end_task(self, dataset):
        # reduce coreset
        if self.task > 0:
            examples_per_class = self.args.buffer_size // ((self.task + 1) * self.cpt)
            buf_x, buf_lab = self.buffer.get_all_data()
            self.buffer.empty()
            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab  = buf_x[idx], buf_lab[idx]
                first = min(ex.shape[0], examples_per_class)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels = lab[:first]
                )
        
        # add new task
        examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
        examples_per_class = examples_last_task // self.cpt
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

        with torch.no_grad():
            for data in dataset.train_loader:
                _, labels, not_aug_inputs = data
                not_aug_inputs = not_aug_inputs.to(self.device)
                if all(ce == 0):
                    break

                flags = torch.zeros(len(labels)).bool()
                for j in range(len(flags)):
                    if ce[labels[j] % self.cpt] > 0:
                        flags[j] = True
                        ce[labels[j] % self.cpt] -= 1

                self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags])
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        losses = self.loss(outputs, labels, reduction='none')
        loss = losses.mean()

        loss.backward()
        self.opt.step()


        return loss.item()
