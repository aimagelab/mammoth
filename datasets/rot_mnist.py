# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone.MNISTMLP import MNISTMLP
from datasets.perm_mnist import MyMNIST, MNIST
from datasets.transforms.rotation import Rotation
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from utils.conf import base_path


class RotatedMNIST(ContinualDataset):
    """
    The Rotated MNIST dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
    """

    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (28, 28)

    def get_data_loaders(self):
        transform = transforms.Compose((Rotation(), transforms.ToTensor()))

        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        test_dataset = MNIST(base_path() + 'MNIST',
                             train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_epochs():
        return 1
