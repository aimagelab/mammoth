# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from datasets.seq_tinyimagenet import SequentialTinyImagenet


class SequentialTinyImagenet32R(SequentialTinyImagenet):
    """The Sequential TinyImagenet dataset resized to 32x32.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
    """
    NAME = 'seq-tinyimg-r'
    MEAN, STD = [0.4807, 0.4485, 0.3980], [0.2541, 0.2456, 0.2604]
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
