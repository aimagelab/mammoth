"""
Implements the Sequential CUB200 Dataset, as used in `Transfer without Forgetting <https://arxiv.org/abs/2206.00388>`_ (Version with ResNet50 as backbone).
"""

import torch
import torchvision.transforms as transforms
from typing import Tuple

from backbone.ResNetBottleneck import resnet50
from datasets.seq_cub200 import SequentialCUB200, MyCUB200, CUB200
from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import store_masked_loaders
from utils.conf import base_path


class MyCUB200RS(MyCUB200):
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(MyCUB200.IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class SequentialCUB200RS(SequentialCUB200):
    """Sequential CUB200 Dataset. Version with ResNet50 (as in `Transfer without Forgetting`)

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """
    NAME = 'seq-cub200-rs'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200RS.IMG_SIZE, MyCUB200RS.IMG_SIZE)
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TRANSFORM = transforms.Compose([
        transforms.Resize(MyCUB200RS.IMG_SIZE),
        transforms.RandomCrop(MyCUB200RS.IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = MyCUB200RS.TEST_TRANSFORM

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCUB200RS(base_path() + 'CUB200', train=True,
                                   download=True, transform=SequentialCUB200RS.TRANSFORM)
        test_dataset = CUB200(base_path() + 'CUB200', train=False,
                              download=True, transform=SequentialCUB200RS.TEST_TRANSFORM)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200RS.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet50_pt"

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200RS.MEAN, SequentialCUB200RS.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCUB200RS.MEAN, SequentialCUB200RS.STD)
        return transform

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 16

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30
