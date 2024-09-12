import glob
import io
import os
import tarfile
import requests
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from datasets.utils import set_default_from_args
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


idx_to_class_names = {
    0: 'airport_inside',
    1: 'artstudio',
    2: 'auditorium',
    3: 'bakery',
    4: 'bar',
    5: 'bathroom',
    6: 'bedroom',
    7: 'bookstore',
    8: 'bowling',
    9: 'buffet',
    10: 'casino',
    11: 'children_room',
    12: 'church_inside',
    13: 'classroom',
    14: 'cloister',
    15: 'closet',
    16: 'clothingstore',
    17: 'computerroom',
    18: 'concert_hall',
    19: 'corridor',
    20: 'deli',
    21: 'dentaloffice',
    22: 'dining_room',
    23: 'elevator',
    24: 'fastfood_restaurant',
    25: 'florist',
    26: 'gameroom',
    27: 'garage',
    28: 'greenhouse',
    29: 'grocerystore',
    30: 'gym',
    31: 'hairsalon',
    32: 'hospitalroom',
    33: 'inside_bus',
    34: 'inside_subway',
    35: 'jewelleryshop',
    36: 'kindergarden',
    37: 'kitchen',
    38: 'laboratorywet',
    39: 'laundromat',
    40: 'library',
    41: 'livingroom',
    42: 'lobby',
    43: 'locker_room',
    44: 'mall',
    45: 'meeting_room',
    46: 'movietheater',
    47: 'museum',
    48: 'nursery',
    49: 'office',
    50: 'operating_room',
    51: 'pantry',
    52: 'poolinside',
    53: 'prisoncell',
    54: 'restaurant',
    55: 'restaurant_kitchen',
    56: 'shoeshop',
    57: 'stairscase',
    58: 'studiomusic',
    59: 'subway',
    60: 'toystore',
    61: 'trainstation',
    62: 'tv_studio',
    63: 'videostore',
    64: 'waitingroom',
    65: 'warehouse',
    66: 'winecellar'
}


class MyMIT67(Dataset):
    NUM_CLASSES = 67

    def __init__(self, root, train=True, download=True, transform=None,
                 target_transform=None) -> None:
        self.root = os.path.join(base_path(), 'MIT67')
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        if not os.path.exists(self.root) and download:
            print('Downloading MIT67 dataset...')
            if not os.path.exists(self.root):
                os.makedirs(self.root)
            train_images_link = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
            train_labels_link = 'https://web.mit.edu/torralba/www/TrainImages.txt'
            test_images_link = 'https://web.mit.edu/torralba/www/TestImages.txt'
            r = requests.get(train_images_link)
            z = tarfile.open(fileobj=io.BytesIO(r.content))
            z.extractall(root)

            r = requests.get(train_labels_link)
            with open(os.path.join(self.root, 'TrainImages.txt'), 'wb') as f:
                f.write(r.content)

            r = requests.get(test_images_link)
            with open(os.path.join(self.root, 'TestImages.txt'), 'wb') as f:
                f.write(r.content)
            print('MIT67 dataset downloaded')
        else:
            print('MIT67 dataset already downloaded')

        folder_targets = {os.path.basename(f[:-1]): i for i, f in enumerate(sorted(glob.glob(os.path.join(self.root, 'Images/*/'))))}

        train_images_path = os.path.join(self.root, 'TrainImages.txt')
        test_images_path = os.path.join(self.root, 'TestImages.txt')

        if self.train:
            with open(train_images_path) as f:
                paths = f.readlines()
        else:
            with open(test_images_path) as f:
                paths = f.readlines()
        paths = [p.strip() for p in paths]
        self.data = [os.path.join(self.root, 'Images', p) for p in paths]
        self.data = np.array(self.data)
        self.targets = [folder_targets[p.split('/')[0]] for p in paths]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = self.targets[index]
        img = Image.open(self.data[index])
        img = img.convert('RGB')

        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        return img, target, not_aug_img


class SequentialMIT67(ContinualDataset):

    NAME = 'seq-mit67'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 67
    N_CLASSES_PER_TASK = [7] * 7 + [6] * 3
    SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def get_data_loaders(self):
        train_dataset = MyMIT67(base_path() + 'MIT67', train=True,
                                download=True, transform=self.TRANSFORM)
        test_dataset = MyMIT67(base_path() + 'MIT76', train=False,
                               download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = list(idx_to_class_names.values())
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return classes

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']

    @staticmethod
    def get_transform():
        transform = transforms.Compose([transforms.ToPILImage(),
                                        SequentialMIT67.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialMIT67.MEAN, SequentialMIT67.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialMIT67.MEAN, SequentialMIT67.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
