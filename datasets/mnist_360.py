# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from copy import deepcopy
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.MNISTMLP import MNISTMLP
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from datasets.perm_mnist import MyMNIST
from datasets.transforms.rotation import IncrementalRotation
from datasets.utils.continual_dataset import store_masked_loaders
from datasets.utils.gcl_dataset import GCLDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path
from PIL import Image


def custom_collate_unbatch(batch):
    # return custom collate
    return [b.squeeze(0) for b in torch.utils.data._utils.collate.default_collate(batch)]


class MNIST360(torch.utils.data.Dataset):
    N_CLASSES = 9

    def __init__(self, args, is_train=False):
        super().__init__()
        self.dataset = []
        self.remaining_training_items = []
        self.num_rounds = 3
        self.args = args
        self.is_train = is_train
        self.is_over = False
        self.completed_rounds, self.test_class, self.test_iteration = 0, 0, 0

        self.train_classes = [0, 1]
        if is_train:
            self.init_train_loaders()
        else:
            self.init_test_loaders()

        self.reinit()

    def train_next_class(self) -> None:
        """
        Changes the couple of current training classes.
        """
        self.train_classes[0] += 1
        self.train_classes[1] += 1
        if self.train_classes[0] == self.N_CLASSES:
            self.train_classes[0] = 0
        if self.train_classes[1] == self.N_CLASSES:
            self.train_classes[1] = 0

        if self.train_classes[0] == 0:
            self.completed_rounds += 1
            if self.completed_rounds == 3:
                self.is_over = True

        if not self.is_over:
            self.active_train_loaders = [
                self.dataset[self.train_classes[0]].pop(),
                self.dataset[self.train_classes[1]].pop()]
            self.active_remaining_items = [
                self.remaining_training_items[self.train_classes[0]].pop(),
                self.remaining_training_items[self.train_classes[1]].pop()]
            self.total_remaining_items = np.hstack([[len(d.dataset) for d in dls] for dls in self.dataset]).sum()  # -= self.current_items
            self.current_items = np.hstack(self.active_remaining_items).sum()

    def init_train_loaders(self) -> None:
        """
        Initializes the test loader.
        """
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True)
        if self.args.validation:
            test_transform = transforms.ToTensor()
            train_dataset, self.val_dataset = get_train_val(
                train_dataset, test_transform, 'mnist-360')

        for j in range(self.N_CLASSES):
            self.dataset.append([])
            self.remaining_training_items.append([])
            train_mask = np.isin(np.array(train_dataset.targets), [j])
            train_rotation = IncrementalRotation(init_deg=(j - 1) * 60,
                                                 increase_per_iteration=360.0 / train_mask.sum())
            for k in range(self.num_rounds * 2):
                tmp_train_dataset = deepcopy(train_dataset)
                numbers_per_batch = train_mask.sum() // (self.num_rounds * 2) + 1
                tmp_train_dataset.data = tmp_train_dataset.data[
                    train_mask][k * numbers_per_batch:(k + 1) * numbers_per_batch]
                tmp_train_dataset.targets = tmp_train_dataset.targets[
                    train_mask][k * numbers_per_batch:(k + 1) * numbers_per_batch]
                tmp_train_dataset.transform = transforms.Compose(
                    [train_rotation, transforms.ToTensor()])
                self.dataset[-1].append(DataLoader(
                    tmp_train_dataset, batch_size=1, shuffle=True))
                self.remaining_training_items[-1].append(
                    tmp_train_dataset.data.shape[0])

    def init_test_loaders(self) -> None:
        """
        Initializes the test loader.
        """
        if self.args.validation:
            test_dataset = self.val_dataset
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                 train=False, download=True)
        for j in range(self.N_CLASSES):
            tmp_test_dataset = deepcopy(test_dataset)
            test_mask = np.isin(np.array(tmp_test_dataset.targets), [j])
            tmp_test_dataset.data = tmp_test_dataset.data[test_mask]
            tmp_test_dataset.targets = tmp_test_dataset.targets[test_mask]
            test_rotation = IncrementalRotation(
                increase_per_iteration=360.0 / test_mask.sum())
            tmp_test_dataset.transform = transforms.Compose(
                [test_rotation, transforms.ToTensor()])
            self.dataset.append(DataLoader(tmp_test_dataset,
                                           batch_size=self.args.batch_size, shuffle=True))

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensembles the next examples of the current classes in a single batch.
        :return: the augmented and not aumented version of the examples of the
                 current batch, along with their labels.
        """
        batch_size_0 = min(int(round(self.active_remaining_items[0] /
                                     (self.active_remaining_items[0] +
                                      self.active_remaining_items[1]) *
                                     self.args.batch_size)),
                           self.active_remaining_items[0])

        batch_size_1 = min(self.args.batch_size - batch_size_0,
                           self.active_remaining_items[1])

        x_train, y_train, x_train_naug = [], [], []
        for j in range(batch_size_0):
            i_x_train, i_y_train, i_x_train_naug = next(iter(
                self.active_train_loaders[0]))
            x_train.append(i_x_train)
            y_train.append(i_y_train)
            x_train_naug.append(i_x_train_naug)
        for j in range(batch_size_1):
            i_x_train, i_y_train, i_x_train_naug = next(iter(
                self.active_train_loaders[1]))
            x_train.append(i_x_train)
            y_train.append(i_y_train)
            x_train_naug.append(i_x_train_naug)
        x_train, y_train, x_train_naug = torch.cat(x_train),\
            torch.cat(y_train), torch.cat(x_train_naug)

        self.active_remaining_items[0] -= batch_size_0
        self.active_remaining_items[1] -= batch_size_1

        if self.active_remaining_items[0] <= 0 or \
                self.active_remaining_items[1] <= 0:
            self.train_next_class()

        return x_train, y_train, x_train_naug

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensembles the next examples of the current class in a batch.
        :return: the batch of examples along with its label.
        """
        x_test, y_test = next(iter(self.dataset[self.test_class]))
        residual_items = len(self.dataset[self.test_class].dataset) - \
            self.test_iteration * self.args.batch_size - len(x_test)
        self.test_iteration += 1
        if residual_items <= 0:
            if residual_items < 0:
                x_test = x_test[:residual_items]
                y_test = y_test[:residual_items]
            self.test_iteration = 0
            self.test_class += 1
            if self.test_class == self.N_CLASSES:
                self.is_over = True

        self.active_remaining_items[0] -= x_test.shape[0]
        return x_test, y_test

    def reinit(self):
        self.is_over = False
        self.completed_rounds, self.test_class, self.test_iteration = 0, 0, 0

        self.train_classes = [0, 1]
        if self.is_train:
            self.active_train_loaders = [
                self.dataset[self.train_classes[0]].pop(),
                self.dataset[self.train_classes[1]].pop()]

            self.active_remaining_items = [
                self.remaining_training_items[self.train_classes[0]].pop(),
                self.remaining_training_items[self.train_classes[1]].pop()]

            self.current_items = np.hstack(self.active_remaining_items).sum()
            self.total_remaining_items = np.hstack([[len(d.dataset) for d in dls] for dls in self.dataset]).sum()
        else:
            self.total_remaining_items = sum([len(d.dataset) for d in self.dataset])
            self.current_items = self.total_remaining_items
            self.active_remaining_items = [self.total_remaining_items]

    def __len__(self):
        clen = self.total_remaining_items - self.current_items + np.hstack(self.active_remaining_items).sum()
        return clen // self.args.batch_size

    def __iter__(self):
        self.reinit()

        return self

    def __getitem__(self, index):
        return next(self)

    def __next__(self):
        if self.is_over:
            raise StopIteration

        if self.is_train:
            return self.get_train_data()
        else:
            return self.get_test_data()


class SequentialMNIST360(GCLDataset):
    """
    MNIST-360 general continual dataset.
    """
    NAME = 'mnist-360'
    SETTING = 'general-continual'
    N_CLASSES = 9
    TRANSFORM = torch.nn.Identity()
    SIZE = (28, 28)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.args = args
        assert args.label_perc == 1, "MNIST-360 does not support partial labels."

    def get_data_loaders(self):
        train_dataset = MNIST360(self.args, is_train=True)
        test_dataset = MNIST360(self.args, is_train=False)

        train_loader = DataLoader(train_dataset,
                                  batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_unbatch)  # dataset is already shuffled and batched
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_unbatch)  # dataset already has dataloader
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        return train_loader, test_loader

    @staticmethod
    def get_backbone() -> torch.nn.Module:
        return MNISTMLP(28 * 28, 10)

    @staticmethod
    def get_loss() -> F.cross_entropy:
        return F.cross_entropy

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 16

    @staticmethod
    def get_epochs():
        return 1
