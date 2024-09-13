import io
import json
import logging
import os
import sys
import zipfile
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple
try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except ImportError:
    raise ImportError("Please install the google_drive_downloader package by running: `pip install googledrivedownloader`")

from datasets.utils import set_default_from_args
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class MyEuroSat(Dataset):

    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.split = split
        assert split in ['train', 'test', 'val'], 'Split must be either train, test or val'
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.ToTensor()

        if not os.path.exists(root + '/DONE'):
            print('Preparing dataset...', file=sys.stderr)
            r = requests.get('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(root)
            os.system(f'mv {root}/EuroSAT_RGB/* {root}')
            os.system(f'rmdir {root}/EuroSAT_RGB')

            # create DONE file
            with open(self.root + '/DONE', 'w') as f:
                f.write('')

            # downlaod split file form https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/
            # from "Conditional Prompt Learning for Vision-Language Models", Kaiyang Zhou et al.
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=self.root + '/split.json')

            print('Done', file=sys.stderr)

        self.data_split = pd.DataFrame(json.load(open(self.root + '/split.json', 'r'))[split])
        self.data = self.data_split[0].values
        self.targets = self.data_split[1].values

        self.class_names = self.get_class_names()

    @staticmethod
    def get_class_names():
        if not os.path.exists(base_path() + f'eurosat/DONE'):
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=base_path() + 'eurosat/split.json')
        return pd.DataFrame(json.load(open(base_path() + 'eurosat/split.json', 'r'))['train'])[2].unique()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(self.root + '/' + img).convert('RGB')

        not_aug_img = self.totensor(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split != 'train':
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


def my_collate_fn(batch):
    tmp = list(zip(*batch))
    imgs = torch.stack(tmp[0], dim=0)
    labels = torch.tensor(tmp[1])
    if len(tmp) == 2:
        return imgs, labels
    not_aug_imgs = tmp[2]
    not_aug_imgs = torch.stack(not_aug_imgs, dim=0)
    if len(tmp) == 4:
        logits = torch.stack(tmp[3], dim=0)
        return imgs, labels, not_aug_imgs, logits
    return imgs, labels, not_aug_imgs


class SequentialEuroSatRgb(ContinualDataset):

    NAME = 'seq-eurosat-rgb'
    SETTING = 'class-il'
    N_TASKS = 5
    N_CLASSES = 10
    N_CLASSES_PER_TASK = 2
    SIZE = (224, 224)
    MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),  # from https://github.dev/KaiyangZhou/Dassl.pytorch defaults
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(SIZE[0], interpolation=InterpolationMode.BICUBIC),  # bicubic
        transforms.CenterCrop(SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        try:
            classes = MyEuroSat.get_class_names()
        except BaseException:
            logging.info("dataset not loaded yet -- loading dataset...")
            MyEuroSat(base_path() + 'eurosat', train=True,
                                    transform=None)
            classes = MyEuroSat.get_class_names()

        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    def get_data_loaders(self):
        train_dataset = MyEuroSat(base_path() + 'eurosat', split='train',
                                  transform=self.TRANSFORM)
        test_dataset = MyEuroSat(base_path() + 'eurosat', split='test',
                                 transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose([transforms.ToPILImage(),
                                        SequentialEuroSatRgb.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialEuroSatRgb.MEAN, std=SequentialEuroSatRgb.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialEuroSatRgb.MEAN, SequentialEuroSatRgb.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    @staticmethod
    def get_prompt_templates():
        return templates['eurosat']
