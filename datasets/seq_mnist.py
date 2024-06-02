# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST

from backbone.MNISTMLP import MNISTMLP
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class SequentialMNIST(ContinualDataset):
    """The Sequential MNIST dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
    """

    NAME = 'seq-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (28, 28)
    TRANSFORM = None

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        test_dataset = MNIST(base_path() + 'MNIST',
                             train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, SequentialMNIST.N_TASKS
                        * SequentialMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 64

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 1
