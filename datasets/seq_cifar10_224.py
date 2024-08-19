# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10, base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates


class SequentialCIFAR10224(ContinualDataset):
    """Sequential CIFAR10 Dataset. The images are resized to 224x224.
    Version with ViT backbone.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-cifar10-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    SIZE = (224, 224)
    TRANSFORM = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomCrop(224, padding=28),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10224.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10224.MEAN, SequentialCIFAR10224.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR10224.MEAN, SequentialCIFAR10224.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR10(base_path() + 'CIFAR10', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']
