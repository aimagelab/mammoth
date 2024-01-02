import os
from requests import request
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
import torch.nn as nn
import yaml
import pickle
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class MyImagenetR(Dataset):
    N_CLASSES = 200

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor()])

        if not os.path.exists(self.root):
            if download:
                # download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
                print("Downloading imagenet-r dataset...")
                url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
                r = request('GET', url, allow_redirects=True)
                if not os.path.exists(self.root):
                    os.makedirs(self.root)
                print("Saving tar...")
                open(self.root + 'imagenet-r.tar', 'wb').write(r.content)
                print("Extracting tar...")
                os.system('tar -xf ' + self.root + 'imagenet-r.tar -C ' + self.root.rstrip('imagenet-r'))

                # move all files in imagenet-r to root with shutil
                import shutil
                print("Moving files...")
                for d in os.listdir(self.root + 'imagenet-r'):
                    shutil.move(self.root + 'imagenet-r/' + d, self.root)

                print("Cleaning up...")
                os.remove(self.root + 'imagenet-r.tar')
                os.rmdir(self.root + 'imagenet-r')

                print("Done!")
            else:
                raise RuntimeError('Dataset not found.')

        pwd = os.path.dirname(os.path.abspath(__file__))
        if self.train:
            data_config = yaml.load(open(pwd + '/imagenet_r_utils/imagenet-r_train.yaml'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(pwd + '/imagenet_r_utils/imagenet-r_test.yaml'), Loader=yaml.Loader)

        self.data = np.array(data_config['data'])
        self.targets = np.array(data_config['targets'])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
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


class SequentialImagenetR(ContinualDataset):

    NAME = 'seq-imagenet-r'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 200
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    normalize = transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.label_to_class_name = self.get_class_names()

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), self.normalize])

        train_dataset = MyImagenetR(base_path() + 'imagenet-r/', train=True,
                                    download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = MyImagenetR(base_path() + 'imagenet-r/', train=False,
                                       download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/imagenet_r_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = label_to_class_name.values()
        class_names = [x.replace('_', ' ') for x in class_names]
        if hasattr(self.args, 'class_order'):
            class_names = [class_names[i] for i in self.class_order]
        return class_names

    @staticmethod
    def get_prompt_templates():
        return templates['imagenet']

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenetR.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        backbone = resnet18()
        num_classes = SequentialImagenetR.N_CLASSES_PER_TASK * SequentialImagenetR.N_TASKS
        backbone.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0, 0, 0),
                                (1, 1, 1))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_virtual_bn_num():
        return 4

    @staticmethod
    def get_n_epochs_first_stage():
        return 50
