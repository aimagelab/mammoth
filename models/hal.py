# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import numpy as np
import torch
from datasets import get_dataset
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--hal_lambda', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)

    return parser


class HAL(ContinualModel):
    NAME = 'hal'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(HAL, self).__init__(backbone, loss, args, transform)
        self.task_number = 0
        self.buffer = Buffer(self.args.buffer_size, self.device, get_dataset(args).N_TASKS, mode='ring')
        self.hal_lambda = args.hal_lambda
        self.beta = args.beta
        self.gamma = args.gamma
        self.anchor_optimization_steps = 100
        self.finetuning_epochs = 1
        self.dataset = get_dataset(args)
        self.spare_model = self.dataset.get_backbone()
        self.spare_model.to(self.device)
        self.spare_opt = SGD(self.spare_model.parameters(), lr=self.args.lr)

    def end_task(self, dataset):
        self.task_number += 1
        # ring buffer mgmt (if we are not loading
        if self.task_number > self.buffer.task_number:
            self.buffer.num_seen_examples = 0
            self.buffer.task_number = self.task_number
        # get anchors (provided that we are not loading the model
        if len(self.anchors) < self.task_number * dataset.N_CLASSES_PER_TASK:
            self.get_anchors(dataset)
            del self.phi

    def get_anchors(self, dataset):
        theta_t = self.net.get_params().detach().clone()
        self.spare_model.set_params(theta_t)

        # fine tune on memory buffer
        for _ in range(self.finetuning_epochs):
            inputs, labels = self.buffer.get_data(self.args.batch_size, transform=self.transform)
            self.spare_opt.zero_grad()
            out = self.spare_model(inputs)
            loss = self.loss(out, labels)
            loss.backward()
            self.spare_opt.step()

        theta_m = self.spare_model.get_params().detach().clone()

        classes_for_this_task = np.unique(dataset.train_loader.dataset.targets)

        for a_class in classes_for_this_task:
            e_t = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            e_t_opt = SGD([e_t], lr=self.args.lr)
            print(file=sys.stderr)
            for i in range(self.anchor_optimization_steps):
                e_t_opt.zero_grad()
                cum_loss = 0

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_m.detach().clone())
                loss = -torch.sum(self.loss(self.spare_model(e_t.unsqueeze(0)), torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_t.detach().clone())
                loss = torch.sum(self.loss(self.spare_model(e_t.unsqueeze(0)), torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                loss = torch.sum(self.gamma * (self.spare_model(e_t.unsqueeze(0), returnt='features') - self.phi) ** 2)
                assert not self.phi.requires_grad
                loss.backward()
                cum_loss += loss.item()

                e_t_opt.step()

            e_t = e_t.detach()
            e_t.requires_grad = False
            self.anchors = torch.cat((self.anchors, e_t.unsqueeze(0)))
            del e_t
            print('Total anchors:', len(self.anchors), file=sys.stderr)

        self.spare_model.zero_grad()

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        if not hasattr(self, 'input_shape'):
            self.input_shape = inputs.shape[1:]
        if not hasattr(self, 'anchors'):
            self.anchors = torch.zeros(tuple([0] + list(self.input_shape))).to(self.device)
        if not hasattr(self, 'phi'):
            print('Building phi', file=sys.stderr)
            with torch.no_grad():
                self.phi = torch.zeros_like(self.net(inputs[0].unsqueeze(0), returnt='features'), requires_grad=False)
            assert not self.phi.requires_grad

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        old_weights = self.net.get_params().detach().clone()

        self.opt.zero_grad()
        outputs = self.net(inputs)

        k = self.task_number

        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        first_loss = 0

        assert len(self.anchors) == self.dataset.N_CLASSES_PER_TASK * k

        if len(self.anchors) > 0:
            first_loss = loss.item()
            with torch.no_grad():
                pred_anchors = self.net(self.anchors)

            self.net.set_params(old_weights)
            pred_anchors -= self.net(self.anchors)
            loss = self.hal_lambda * (pred_anchors ** 2).mean()
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            self.phi = self.beta * self.phi + (1 - self.beta) * self.net(inputs[:real_batch_size], returnt='features').mean(0)

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return first_loss + loss.item()
