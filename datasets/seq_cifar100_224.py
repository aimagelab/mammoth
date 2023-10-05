

import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
# from backbone.ResNet18 import resnet18_twf1
import torch.nn.functional as F
import numpy as np
from utils.conf import base_path_dataset as base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torchvision
import torch.nn as nn
from datasets.seq_cifar100 import MyCIFAR100, TCIFAR100
from timm import create_model


class SequentialCIFAR100224(ContinualDataset):

    NAME = 'seq-cifar100-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = 100
    SIZE = (224, 224)
    MEAN, STD = (0, 0, 0), (1, 1, 1)  # Normalized in [0,1] as in L2P paper
    TRANSFORM = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomCrop(224, padding=28),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)]
    )
    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                     download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100224.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        model_name = 'vit_base_patch16_224'
        return create_model(
            model_name,
            pretrained=True,
            num_classes=SequentialCIFAR100224.N_CLASSES
        )

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @staticmethod
    def get_epochs():
        return 5

    @staticmethod
    def get_batch_size():
        return 128
