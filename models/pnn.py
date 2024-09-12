# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.args import ArgumentParser
from utils.conf import get_device
from models.utils.continual_model import ContinualModel
from backbone import get_backbone


def get_pnn_backbone(bone, old_cols=None, x_shape=None):
    from backbone.MNISTMLP import BaseMNISTMLP
    from backbone.MNISTMLP_PNN import MNISTMLP_PNN
    from backbone.ResNetBlock import ResNet
    from backbone.ResNet18_PNN import resnet18_pnn

    if isinstance(bone, BaseMNISTMLP):
        return MNISTMLP_PNN(bone.input_size, bone.output_size, old_cols)
    elif isinstance(bone, ResNet):
        return resnet18_pnn(bone.num_classes, bone.nf, old_cols, x_shape)
    else:
        raise NotImplementedError('Progressive Neural Networks is not implemented for this backbone')


class Pnn(ContinualModel):
    """Progressive Neural Networks."""
    NAME = 'pnn'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        self.nets = [get_pnn_backbone(backbone).to(get_device())]
        backbone = self.nets[-1]
        super(Pnn, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.x_shape = None
        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.task_idx = 0

    def forward(self, x, task_label):
        if self.x_shape is None:
            self.x_shape = x.shape

        start_idx, end_idx = self.dataset.get_offsets(task_label)
        if self.task_idx == 0:
            out = self.net(x)
        else:
            self.nets[task_label].to(self.device)
            out = self.nets[task_label](x)
            if self.task_idx != task_label:
                self.nets[task_label].cpu()

        # mask out previous tasks - Task-IL forward
        if start_idx > 0:
            out[:, :start_idx] = -torch.inf
        out[:, end_idx:] = -torch.inf
        return out

    def end_task(self, dataset):
        # instantiate new column
        self.task_idx += 1
        self.nets[-1].cpu()
        self.nets.append(get_pnn_backbone(get_backbone(self.args), self.nets, self.x_shape).to(self.device))
        self.net = self.nets[-1]
        self.opt = self.get_optimizer()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.x_shape is None:
            self.x_shape = inputs.shape

        self.net.to(self.device)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
