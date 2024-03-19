# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets.transforms.driftTransforms import DefocusBlur, GaussianNoise, JpegCompression, ShotNoise, SpeckleNoise
import copy


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError(
                'The dataset must be initialized with all the required fields.')

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError


def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # selecting previous the training classes to apply drift
    train_drift_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK,
                                      np.array(train_dataset.targets) < setting.i)

    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    # selecting the previous test class to apply drift
    test_drift_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK,
                                     np.array(test_dataset.targets) < setting.i)

    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                               np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    # selecting the unseen second half of the previous class to apply drift
    drifting_train_dataset = copy.deepcopy(train_dataset)
    drifting_train_dataset.data = drifting_train_dataset.data[train_drift_mask]
    drifting_train_dataset.data = drifting_train_dataset.data[len(drifting_train_dataset.data)//2:]
    drifting_train_dataset.targets = np.array(drifting_train_dataset.targets)[train_drift_mask]
    drifting_train_dataset.targets = drifting_train_dataset.targets[len(drifting_train_dataset.targets)//2:]

    # selecting half of the task classes for training and leaving the rest for drift in the next iteration
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.data = train_dataset.data[: len(train_dataset.data)//2]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    train_dataset.targets = train_dataset.targets[: len(train_dataset.targets)//2]

    drifting_test_dataset = copy.deepcopy(test_dataset)
    drifting_test_dataset.data = drifting_test_dataset.data[test_drift_mask]
    drifting_test_dataset.targets = np.array(drifting_test_dataset.targets)[test_drift_mask]

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    DRIFT_SEVERITY = 5

    DRIFTS = [DefocusBlur(DRIFT_SEVERITY),
              GaussianNoise(DRIFT_SEVERITY),
              JpegCompression(DRIFT_SEVERITY),
              ShotNoise(DRIFT_SEVERITY),
              SpeckleNoise(DRIFT_SEVERITY)]

    TRAIN_DRIFT_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        DRIFTS[3],  # drift applied to training data
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2615))
    ])

    combined_train_dataset = []

    for img, target, not_aug_img in drifting_train_dataset:
        img = TRAIN_DRIFT_TRANSFORM(img)
        combined_train_dataset.append((img, target, not_aug_img))

    for img, target, not_aug_img in train_dataset:
        img = setting.TRANSFORM(img)
        combined_train_dataset.append((img, target, not_aug_img))

    train_loader = DataLoader(combined_train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)

    setting.train_loader = train_loader

    TEST_DRIFT_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        DRIFTS[3],  # drift applied to test data
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2615))
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2615))])

    if setting.i > 0:
        drifted_test_dataset = []

        for img, target in drifting_test_dataset:
            img = TEST_DRIFT_TRANSFORM(img)
            drifted_test_dataset.append((img, target))

        drifted_test_loader = DataLoader(drifted_test_dataset,
                                         batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
        
        # replacing the previous test loader with drifted images
        assert len(setting.test_loaders) > 0
        setting.test_loaders[len(setting.test_loaders) - 1] = drifted_test_loader

    current_test_dataset = []

    for img, target in test_dataset:
        img = TEST_TRANSFORM(img)
        current_test_dataset.append((img, target))

    test_loader = DataLoader(current_test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)

    # current test loader contains undrifted images
    setting.test_loaders.append(test_loader)

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: Dataset, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i -
                                setting.N_CLASSES_PER_TASK, np.array(
                                    train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
