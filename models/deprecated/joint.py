"""
This module contains the implementation of the Joint CL model.
"""

import math
import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.conf import create_seeded_dataloader
from utils.status import progress_bar


def get_parser() -> ArgumentParser:
    """
    Returns the ArgumentParser object for the joint model.
    """
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    """
    The Joint CL model. The model is deprecated, use the option `--joint=1` instead combined with the SGD model.

    Attributes:
        NAME (str): joint.
        COMPATIBILITY (list): the joint model is compabible with class-il, domain-il and task-il scenarios. For a joint model for the general-continual scenario, see the joint_gcl model.
    """

    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):

        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset):
        """
        This version of joint training simply saves all data from previous tasks and then trains on all data at the end of the last one.
        """
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = self.get_optimizer()

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = create_seeded_dataloader(self.args, temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i + 1) * bs]
                    labels = all_labels[order][i * bs: (i + 1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        This version of joint training does nothing during incremental CL training.
        """
        return 0
