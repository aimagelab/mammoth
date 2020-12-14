# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from tqdm import tqdm
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class JointGCL(ContinualModel):
    NAME = 'joint_gcl'
    COMPATIBILITY = ['general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(JointGCL, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset):
      # reinit network
      self.net = dataset.get_backbone()
      self.net.to(self.device)
      self.net.train()
      self.opt = SGD(self.net.parameters(), lr=self.args.lr)

      # gather data
      all_data = torch.cat(self.old_data)
      all_labels = torch.cat(self.old_labels)

      # train
      for e in range(1):#range(self.args.n_epochs):
        rp = torch.randperm(len(all_data))
        for i in range(math.ceil(len(all_data) / self.args.batch_size)):
            inputs = all_data[rp][i * self.args.batch_size:(i+1) * self.args.batch_size]
            labels = all_labels[rp][i * self.args.batch_size:(i+1) * self.args.batch_size]
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            progress_bar(i, math.ceil(len(all_data) / self.args.batch_size), e, 'J', loss.item())

    def observe(self, inputs, labels, not_aug_inputs):
        self.old_data.append(inputs.data)
        self.old_labels.append(labels.data)
        return 0
