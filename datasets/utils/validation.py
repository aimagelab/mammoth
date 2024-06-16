# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from utils import create_if_not_exists


class ValidationDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: np.ndarray,
                 transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            if np.max(img) < 2:
                img = Image.fromarray(np.uint8(img * 255))
            else:
                img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_validation_indexes(validation_size: float, dataset: Dataset, seed=None) -> Tuple[Dataset, Dataset]:
    """
    Returns the indexes of train and validation datasets from the given dataset, according to the validation size.
    
    Args:
        validation_size (float): percentage of samples for each class to be used for validation (between 0 and 100)
        dataset (Dataset): the dataset to split
        seed (int): the seed for the random generator. If None, the seed is set to 0

    Returns:
        tuple: the train and validation dataset indexes
    """
    seed = 0 if seed is None else seed
    if validation_size < 1:
        validation_size = round(validation_size * 100, 2)

    cls_ids, samples_per_class = np.unique(dataset.targets, return_counts=True)
    n_samples_val_per_class = np.ceil(samples_per_class * (validation_size / 100)).astype(int)

    all_idxs = np.arange(len(dataset.targets))
    val_idxs, train_idxs = [], []
    for cls_id, n_samples, n_samples_val in zip(cls_ids, samples_per_class, n_samples_val_per_class):
        cls_idxs = all_idxs[dataset.targets == cls_id]
        idxs = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed)).numpy()
        val_idxs.append(cls_idxs[idxs[:n_samples_val]])
        train_idxs.append(cls_idxs[idxs[n_samples_val:]])
        
    train_idxs = np.concatenate(train_idxs)
    val_idxs = np.concatenate(val_idxs)

    return train_idxs, val_idxs

def get_train_val(train: Dataset, test_transform: nn.Module,
                  dataset: str, val_perc: float = 0.1):
    """
    Extract val_perc% of the training set as the validation set.

    Args:
        train: training dataset
        test_transform: transformation of the test dataset
        dataset: dataset name
        val_perc: percentage of the training set to be extracted

    Returns:
        the training set and the validation set
    """
    dataset_length = train.data.shape[0]
    directory = 'datasets/val_permutations/'
    create_if_not_exists(directory)
    file_name = dataset + '.pt'
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length)
        torch.save(perm, directory + file_name)

    train_idxs, val_idxs = get_validation_indexes(val_perc, train)

    test_dataset = ValidationDataset(train.data[val_idxs],
                                     train.targets[val_idxs],
                                     transform=test_transform)
    train.data = train.data[train_idxs]
    train.targets = train.targets[train_idxs]

    return train, test_dataset
