# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from backbone.MNISTMLP import MNISTMLP
from datasets.perm_mnist import MyMNIST
from datasets.transforms.rotation import IncrementalRotation
from datasets.utils.gcl_dataset import GCLDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path, create_seeded_dataloader
from datasets.utils import set_default_from_args


def custom_collate_unbatch(batch) -> List[torch.Tensor]:
    """
    Custom collate function to unbatch a batch of data.

    Args:
        batch (list): A list of tensors representing a batch of data.

    Returns:
        list: A list of tensors, where each tensor is unbatched from the input batch.
    """
    return [b.squeeze(0) for b in torch.utils.data._utils.collate.default_collate(batch)]


class MNIST360(torch.utils.data.Dataset):
    """A custom dataset class for MNIST360 that provides training and testing data
    with incremental rotation for each class.

    Args:
        args (object): An object containing the arguments for the dataset.
        is_train (bool): A flag indicating whether the dataset is for training or testing.

    Attributes:
        N_CLASSES (int): The number of classes in the dataset.
        dataset (list): A list of data loaders for each class.
        remaining_training_items (list): A list of the remaining training items for each class.
        num_rounds (int): The number of rounds for each class.
        args (object): An object containing the arguments for the dataset.
        is_train (bool): A flag indicating whether the dataset is for training or testing.
        is_over (bool): A flag indicating whether the dataset is completed.
        completed_rounds (int): The number of completed rounds.
        test_class (int): The current test class index.
        test_iteration (int): The current test iteration index.
        train_classes (list): A list of the current training classes.
        active_train_loaders (list): A list of the active training data loaders.
        current_items (int): The current number of items in the dataset.
    """

    N_CLASSES = 9

    def __init__(self, args: Namespace, is_train: bool = False) -> None:
        super().__init__()
        self.num_rounds = 3
        self.args = args
        self.is_train = is_train

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

    def init_train_loaders(self) -> None:
        """
        Initializes the train loader.
        """
        self.remaining_training_items = []
        self.dataset = []

        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True)
        if self.args.validation:
            test_transform = transforms.ToTensor()
            train_dataset, self.val_dataset = get_train_val(
                train_dataset, test_transform, 'mnist-360', val_perc=self.args.validation / 100)

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
                self.dataset[-1].append(create_seeded_dataloader(self.args,
                                                                 tmp_train_dataset, batch_size=1, shuffle=True, num_workers=0))
                self.remaining_training_items[-1].append(
                    tmp_train_dataset.data.shape[0])

    def init_test_loaders(self) -> None:
        """
        Initializes the test loader.
        """
        self.remaining_training_items = []
        self.dataset = []

        if self.args.validation:
            test_transform = transforms.ToTensor()
            train_dataset = MyMNIST(base_path() + 'MNIST',
                                    train=True, download=True)
            _, test_dataset = get_train_val(
                train_dataset, test_transform, 'mnist-360', val_perc=self.args.validation)
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
            self.dataset.append(create_seeded_dataloader(self.args, tmp_test_dataset,
                                                         batch_size=self.args.batch_size, shuffle=False, num_workers=0))

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensembles the next examples of the current classes in a single batch.

        Returns:
            Tensor: The batch of examples.

            Tensor: The labels of the examples.

            Tensor: The batch of examples without augmentation.
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
        x_train, y_train, x_train_naug = torch.cat(x_train), \
            torch.cat(y_train), torch.cat(x_train_naug)

        self.active_remaining_items[0] -= batch_size_0
        self.active_remaining_items[1] -= batch_size_1

        if self.active_remaining_items[0] <= 0 or \
                self.active_remaining_items[1] <= 0:
            self.train_next_class()

        return x_train, y_train, x_train_naug

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensembles the next examples of the current class in a batch.

        Returns:
            Tensor: The batch of examples.
            Tensor: The labels of the examples.
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

        return x_test, y_test

    def reinit(self) -> None:
        self.is_over = False
        self.completed_rounds, self.test_class, self.test_iteration = 0, 0, 0

        self.train_classes = [0, 1]

        if self.is_train:
            self.init_train_loaders()
        else:
            self.init_test_loaders()

        if self.is_train:
            self.active_train_loaders = [
                self.dataset[self.train_classes[0]].pop(),
                self.dataset[self.train_classes[1]].pop()]

            self.active_remaining_items = [
                self.remaining_training_items[self.train_classes[0]].pop(),
                self.remaining_training_items[self.train_classes[1]].pop()]

    def __iter__(self):
        self.reinit()

        return self

    def __next__(self):
        if self.is_over:
            raise StopIteration

        if self.is_train:
            return self.get_train_data()
        else:
            return self.get_test_data()


class SequentialMNIST360(GCLDataset):
    """
    A dataset class for the MNIST-360 dataset in the context of general-continual learning.

    Attributes:
        NAME (str): The name of the dataset.
        SETTING (str): The setting of the dataset.
        N_CLASSES (int): The number of classes in the dataset.
        TRANSFORM (torch.nn.Module): The transformation to apply to the data.
        SIZE (tuple): The size of the input images.
        args (Namespace): An object containing the arguments for the dataset.
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

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get the data loaders for the MNIST360 dataset, add them to the current object and return them.

        Returns:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.

            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        """
        train_dataset = MNIST360(self.args, is_train=True)
        test_dataset = MNIST360(self.args, is_train=False)

        # dataset is already shuffled and batched internally - no need for a dataloader
        self.test_loaders.append(test_dataset)
        self.train_loader = train_dataset

        return train_dataset, test_dataset

    @staticmethod
    def get_backbone() -> torch.nn.Module:
        return MNISTMLP(28 * 28, 10)

    @staticmethod
    def get_loss() -> Callable:
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

    @set_default_from_args('batch_size')
    def get_batch_size(self) -> int:
        return 16

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 1


if __name__ == "__main__":
    ds = SequentialMNIST360(Namespace(validation=False, label_perc=1, n_epochs=1, batch_size=16, permute_classes=False, joint=False, num_workers=0, seed=None))
    train, test = ds.get_data_loaders()

    # load all data and save it in results/tmp
    import os
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    os.makedirs('../data/results/mnist360images/tmp/train', exist_ok=True)
    for i, (x, y, _) in tqdm(enumerate(train), total=len(train)):
        torchvision.utils.save_image(x, f'../data/results/mnist360images/tmp/train/{i}.png')

    os.makedirs('../data/results/mnist360images/tmp/test', exist_ok=True)
    for i, (x, y) in tqdm(enumerate(test), total=len(test)):
        torchvision.utils.save_image(x, f'../data/results/mnist360images/tmp/test/{i}.png')
