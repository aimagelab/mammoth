import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from typing import Tuple

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class ChestX(Dataset):
    N_CLASSES = 6

    """
    To reduce the effect of the severe imbalance in the dataset, we drop the two classes with the smallest and largest amount of samples.
    """
    LABELS = [
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Fibrosis",
        "Pleural Thickening",
        "Pneumothorax"
    ]

    """
    Overrides the ChestX dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(f'{root}/train_images.pkl'):
            if download:
                from onedrivedownloader import download

                print('Downloading dataset')
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EfmFCiLaGlpFgtAuv0YLpeYBeR54I7YHK75bu_Ex78mADA?e=K8rHpZ"
                download(ln, filename=smart_joint(root, 'chestx.zip'), unzip=True, unzip_path=root.rstrip('chestx'), clean=True)
            else:
                raise FileNotFoundError(f'File not found: {root}/train_images.pkl')

        if train:
            filename_labels = f'{self.root}/train_labels.pkl'
            filename_images = f'{self.root}/train_images.pkl'
        else:
            filename_labels = f'{self.root}/test_labels.pkl'
            filename_images = f'{self.root}/test_images.pkl'

        self.not_aug_transform = transforms.ToTensor()

        with open(filename_images, 'rb') as f:
            self.data = pickle.load(f)

        with open(filename_labels, 'rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = np.repeat(img[np.newaxis, :, :], 3, axis=0)
        img = Image.fromarray((img * 255).astype(np.int8).transpose(1, 2, 0), mode='RGB')

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialChestX(ContinualDataset):

    NAME = 'seq-chestx'
    SETTING = 'class-il'
    N_TASKS = 2
    N_CLASSES = 6
    N_CLASSES_PER_TASK = 3
    SIZE = (224, 224)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=MEAN,
                                     std=STD)

    TRANSFORM = transforms.Compose([
        transforms.Resize(size=SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    def get_data_loaders(self):
        train_dataset = ChestX(base_path() + 'chestx', train=True,
                               download=True, transform=self.TRANSFORM)

        test_dataset = ChestX(base_path() + 'chestx', train=False, download=True,
                              transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(ChestX.LABELS, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(),
                                   SequentialChestX.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialChestX.MEAN, std=SequentialChestX.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean=SequentialChestX.MEAN, std=SequentialChestX.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
