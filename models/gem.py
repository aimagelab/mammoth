# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the gem_license file in the root of this source tree.

import quadprog

import numpy as np
import torch
from models.utils.continual_model import ContinualModel

from utils.buffer import Buffer
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # remove minibatch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'minibatch_size':
            del parser._actions[i]
            break

    parser.add_argument('--gamma', type=float, default=None,
                        help='Margin parameter for GEM.')
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


class Gem(ContinualModel):
    NAME = 'gem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Gem, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Allocate temporary synaptic memory
        self.grad_dims = []
        for pp in self.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        self.current_task += 1
        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))

        # add data to the buffer
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device),
            task_labels=torch.ones(samples_per_task,
                dtype=torch.long).to(self.device) * (self.current_task - 1)
        )


    def observe(self, inputs, labels, not_aug_inputs):

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.buffer_size, transform=self.transform)

            for tt in buf_task_labels.unique():
                # compute gradient on the memory buffer
                self.opt.zero_grad()
                cur_task_inputs = buf_inputs[buf_task_labels == tt]
                cur_task_labels = buf_labels[buf_task_labels == tt]
                cur_task_outputs = self.forward(cur_task_inputs)
                penalty = self.loss(cur_task_outputs, cur_task_labels)
                penalty.backward()
                store_grad(self.parameters, self.grads_cs[tt], self.grad_dims)

        # now compute the grad on the current data
        self.opt.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        # check if gradient violates buffer constraints
        if not self.buffer.is_empty():
            # copy gradient
            store_grad(self.parameters, self.grads_da, self.grad_dims)

            dot_prod = torch.mm(self.grads_da.unsqueeze(0),
                            torch.stack(self.grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self.grads_da.unsqueeze(1),
                              torch.stack(self.grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads_da,
                               self.grad_dims)

        self.opt.step()

        return loss.item()
