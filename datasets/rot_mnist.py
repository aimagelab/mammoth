# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.perm_mnist import MyMNIST, MNIST
from datasets.transforms.rotation import Rotation
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args
from torchvision.datasets import MNIST


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

    @set_default_from_args("backbone")
    def get_backbone():
        return "mnistmlp"

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

    @set_default_from_args('batch_size')
    def get_batch_size(self) -> int:
        return 128

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 1

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = MNIST(base_path() + 'MNIST', train=True, download=True).classes
        classes = [c.split('-')[1].strip() for c in classes]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
