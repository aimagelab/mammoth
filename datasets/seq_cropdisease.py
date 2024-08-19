import json
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class CropDisease(Dataset):

    LABELS = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry___Powdery_mildew",
        "Cherry___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight",
        "Corn___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()]
        )

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EZUaXKQUAVBPrhjHTUdflDEBNu0YiPWrdpAdDhnEU4nD2A?e=GPrCYF"
                print('Downloading dataset')
                parent_dir = os.path.dirname(root)
                download(ln, filename=os.path.join(root, 'cropdisease.tar.gz'), unzip=True, unzip_path=parent_dir, clean=True)

        filename = smart_joint(root, ('train' if train else 'test') + '.json')
        with open(filename) as f:
            data_config = json.load(f)

        self.data = np.array([smart_joint(root, 'images', d) for d in data_config['data']])
        self.targets = np.array(data_config['labels']).astype(np.int16)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

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


class SequentialCropDisease(ContinualDataset):

    NAME = 'seq-cropdisease'
    SETTING = 'class-il'
    N_TASKS = 7
    N_CLASSES = 35
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    SIZE = (224, 224)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def get_data_loaders(self):
        train_dataset = CropDisease(base_path() + 'cropdisease', train=True,
                                    download=True, transform=self.TRANSFORM)
        test_dataset = CropDisease(base_path() + 'cropdisease', train=False,
                                   download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = [x.replace('_', ' ') for x in CropDisease.LABELS]  # .split('___')[-1]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCropDisease.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialCropDisease.MEAN, std=SequentialCropDisease.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCropDisease.MEAN, SequentialCropDisease.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
