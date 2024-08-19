# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from backbone.ResNetBlock import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils import smart_joint
from utils.conf import base_path
from datasets.utils import set_default_from_args


class TinyImagenet(Dataset):
    """Defines the Tiny Imagenet dataset."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download

                print('Downloading dataset')
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
                download(ln, filename=smart_joint(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(smart_joint(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(smart_joint(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """Overrides the TinyImagenet dataset to change the getitem function."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialTinyImagenet(ContinualDataset):
    """The Sequential Tiny Imagenet dataset.

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

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                       train=True, download=True, transform=transform)
        test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                    train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialTinyImagenet.MEAN, SequentialTinyImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialTinyImagenet.MEAN, SequentialTinyImagenet.STD)
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
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names


CLASS_NAMES = [
    'egyptian_cat',
    'reel',
    'volleyball',
    'rocking_chair',
    'lemon',
    'bullfrog',
    'basketball',
    'cliff',
    'espresso',
    'plunger',
    'parking_meter',
    'german_shepherd',
    'dining_table',
    'monarch',
    'brown_bear',
    'school_bus',
    'pizza',
    'guinea_pig',
    'umbrella',
    'organ',
    'oboe',
    'maypole',
    'goldfish',
    'potpie',
    'hourglass',
    'seashore',
    'computer_keyboard',
    'arabian_camel',
    'ice_cream',
    'nail',
    'space_heater',
    'cardigan',
    'baboon',
    'snail',
    'coral_reef',
    'albatross',
    'spider_web',
    'sea_cucumber',
    'backpack',
    'labrador_retriever',
    'pretzel',
    'king_penguin',
    'sulphur_butterfly',
    'tarantula',
    'lesser_panda',
    'pop_bottle',
    'banana',
    'sock',
    'cockroach',
    'projectile',
    'beer_bottle',
    'mantis',
    'freight_car',
    'guacamole',
    'remote_control',
    'european_fire_salamander',
    'lakeside',
    'chimpanzee',
    'pay-phone',
    'fur_coat',
    'alp',
    'lampshade',
    'torch',
    'abacus',
    'moving_van',
    'barrel',
    'tabby',
    'goose',
    'koala',
    'bullet_train',
    'cd_player',
    'teapot',
    'birdhouse',
    'gazelle',
    'academic_gown',
    'tractor',
    'ladybug',
    'miniskirt',
    'golden_retriever',
    'triumphal_arch',
    'cannon',
    'neck_brace',
    'sombrero',
    'gasmask',
    'candle',
    'desk',
    'frying_pan',
    'bee',
    'dam',
    'spiny_lobster',
    'police_van',
    'ipod',
    'punching_bag',
    'beacon',
    'jellyfish',
    'wok',
    "potter's_wheel",
    'sandal',
    'pill_bottle',
    'butcher_shop',
    'slug',
    'hog',
    'cougar',
    'crane',
    'vestment',
    'dragonfly',
    'cash_machine',
    'mushroom',
    'jinrikisha',
    'water_tower',
    'chest',
    'snorkel',
    'sunglasses',
    'fly',
    'limousine',
    'black_stork',
    'dugong',
    'sports_car',
    'water_jug',
    'suspension_bridge',
    'ox',
    'ice_lolly',
    'turnstile',
    'christmas_stocking',
    'broom',
    'scorpion',
    'wooden_spoon',
    'picket_fence',
    'rugby_ball',
    'sewing_machine',
    'steel_arch_bridge',
    'persian_cat',
    'refrigerator',
    'barn',
    'apron',
    'yorkshire_terrier',
    'swimming_trunks',
    'stopwatch',
    'lawn_mower',
    'thatch',
    'fountain',
    'black_widow',
    'bikini',
    'plate',
    'teddy',
    'barbershop',
    'confectionery',
    'beach_wagon',
    'scoreboard',
    'orange',
    'flagpole',
    'american_lobster',
    'trolleybus',
    'drumstick',
    'dumbbell',
    'brass',
    'bow_tie',
    'convertible',
    'bighorn',
    'orangutan',
    'american_alligator',
    'centipede',
    'syringe',
    'go-kart',
    'brain_coral',
    'sea_slug',
    'cliff_dwelling',
    'mashed_potato',
    'viaduct',
    'military_uniform',
    'pomegranate',
    'chain',
    'kimono',
    'comic_book',
    'trilobite',
    'bison',
    'pole',
    'boa_constrictor',
    'poncho',
    'bathtub',
    'grasshopper',
    'walking_stick',
    'chihuahua',
    'tailed_frog',
    'lion',
    'altar',
    'obelisk',
    'beaker',
    'bell_pepper',
    'bannister',
    'bucket',
    'magnetic_compass',
    'meat_loaf',
    'gondola',
    'standard_poodle',
    'acorn',
    'lifeboat',
    'binoculars',
    'cauliflower',
    'african_elephant'
]
