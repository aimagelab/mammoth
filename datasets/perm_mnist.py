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
from datasets.transforms.permutation import Permutation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path, create_seeded_dataloader


def store_mnist_loaders(transform: transforms.Compose,
                        setting: ContinualDataset) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates the data loaders for the MNIST dataset.

    Args:
        transform: the transformation to apply to the data
        setting: the setting of the experiment

    Returns:
        the training and test data loaders
    """
    train_dataset = MyMNIST(base_path() + 'MNIST',
                            train=True, download=True, transform=transform)
    if setting.args.validation:
        train_dataset, test_dataset = get_train_val(train_dataset,
                                                    transform, setting.NAME)
    else:
        test_dataset = MNIST(base_path() + 'MNIST',
                             train=False, download=True, transform=transform)

    train_loader = create_seeded_dataloader(setting.args, train_dataset,
                                            batch_size=setting.args.batch_size, shuffle=True)
    test_loader = create_seeded_dataloader(setting.args, test_dataset,
                                           batch_size=setting.args.batch_size, shuffle=False)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyMNIST, self).__init__(root, train, transform,
                                      target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        
        Args:
            index: index of the element to be returned
        
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img


class PermutedMNIST(ContinualDataset):
    """Permuted MNIST Dataset.

    Creates a dataset composed by a sequence of tasks, each containing a
    different permutation of the pixels of the MNIST dataset.

    Args:
        NAME (str): name of the dataset
        SETTING (str): setting of the experiment
        N_CLASSES_PER_TASK (int): number of classes in each task
        N_TASKS (int): number of tasks
        SIZE (tuple): size of the images
    """

    NAME = 'perm-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    SIZE = (28, 28)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = transforms.Compose((transforms.ToTensor(), Permutation()))
        train, test = store_mnist_loaders(transform, self)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, PermutedMNIST.N_CLASSES_PER_TASK)

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
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_epochs():
        return 1
