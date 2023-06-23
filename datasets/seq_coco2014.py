
from tqdm import tqdm
import torch
import numpy as np

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    import sys, os
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path = [mammoth_path] + sys.path

from backbone.ResNet18 import resnet18
from PIL import Image

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
import os
import json
import pickle
from typing import Tuple
import h5py
from pathlib import Path

class Coco2014(Dataset):
    def __init__(self, root, train=True, transform=None, task=0):
        assert task in range(4)
        self.root = root if not os.getenv("COCO_METAFILES") else os.getenv("COCO_METAFILES")
        self.transform = transform
        self.train = train

        img_file = Path(self.root) / f'{"test" if not train else "train"}_task{task}_imgs_coco.hdf5'
        multihot_file = Path(self.root) / f'{"test" if not train else "train"}_task{task}_multi_hot_categories_coco.json'

        self.imgs = torch.from_numpy(np.asarray(h5py.File(img_file, 'r')['images']))
        self.multihot_labels = torch.from_numpy(np.asarray(json.load(open(multihot_file, 'r'))))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label_vector = self.multihot_labels[index]        
        original_img = img.copy()

        #not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if not self.train:
            return img, label_vector
        return img, label_vector, original_img


def tags_analisys(tags: dict, plot_hist: bool = False):
    print(f'Number of tags: {len(tags)}')
    print(f'Max frequency: {max(tags.values())}')
    print(f'Min frequency: {min(tags.values())}')
    print(f'Mean frequency: {np.mean(list(tags.values())):.2f}')
    print(f'STD frequency: {np.std(list(tags.values())):.2f}')
    print(f'Median frequency: {np.median(list(tags.values())):.2f}')
    if plot_hist:
        import matplotlib.pyplot as plt
        plt.hist(list(tags.values()), bins=100)
        plt.show()


def get_tags_from_data(data: list):
    tags = {}
    for item in data:
        for tag in item['tags']:
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1
    return tags


def data_tags_analisys(data: list, tag_analisys=True, plot_hist: bool = False):
    tags_len_list = [len(item['tags']) for item in data]
    print(f'Data len: {len(data)}')
    print(f'Mean tags per image: {np.mean(tags_len_list):.2f}')
    print(f'STD tags per image: {np.std(tags_len_list):.2f}')
    print(f'Median tags per image: {np.median(tags_len_list):.2f}')
    print(f'Max tags per image: {max(tags_len_list)}')
    print(f'Min tags per image: {min(tags_len_list)}')
    if plot_hist:
        import matplotlib.pyplot as plt
        plt.hist(tags_len_list, bins=100)
        plt.show()
    if tag_analisys:
        tags = get_tags_from_data(data)
        tags_analisys(tags, plot_hist)

def webvision_collate_fn(batch):
        data = list(zip(*batch))
        inp = torch.stack(data[0], 0)
        tgt = torch.stack(data[1], 0)
        return (inp, tgt, data[2])

def store_webvision_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # datalen = len(train_dataset)
    # datastep = datalen // setting.N_TASKS
    # train_mask = np.logical_and(np.arange(datalen) >= (setting.i * datastep),
    #                             np.arange(datalen) < ((setting.i + 1) * datastep))

    # train_dataset.images_info_list = (np.array(train_dataset.images_info_list)[train_mask]).tolist()

    # print(f'\nTask {setting.i} analisys:')
    # data_tags_analisys(train_dataset.images_info_list)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4, collate_fn=webvision_collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader


class SequentialCoco2014(ContinualDataset):

    NAME = 'seq-coco2014'
    SETTING = 'multi-label'
    N_CLASSES_PER_TASK = -1 # this should not be used
    N_TASKS = 4
    N_CLASSES = 70
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4569, 0.4354, 0.3904),
                             (0.2737, 0.2664, 0.2791)),
    ])
    TEST_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4569, 0.4354, 0.3904),
                             (0.2737, 0.2664, 0.2791)),
    ])

    def get_examples_number(self):
        train_dataset = Coco2014(base_path() + 'coco2014', train=True)
        return len(train_dataset)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            self.get_normalization_transform(),
        ])

        train_dataset = Coco2014(base_path() + 'coco2014', train=True, transform=transform, task=self.i)

        test_dataset = Coco2014(base_path() + 'coco2014', train=False, transform=test_transform, task=self.i)

        train, test = store_webvision_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCoco2014.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        # TODO
        return resnet18(SequentialCoco2014.N_CLASSES)

    @staticmethod
    def get_loss():
        return torch.nn.BCEWithLogitsLoss(reduction='mean')

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4569, 0.4354, 0.3904),
                                         (0.2737, 0.2664, 0.2791))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4569, 0.4354, 0.3904),
                                (0.2737, 0.2664, 0.2791))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCoco2014.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler


if __name__ == '__main__':
    # for testing
    # args = lambda x: x
    # args.batch_size = 32
    # dataset = SequentialWebVision(args)
    # dataset.get_data_loaders()
    # dataset.train_loader.dataset[100]
    # [dataset.get_data_loaders() for _ in range(dataset.N_TASKS)]
    thej = Coco2014('', train=True, task = 0)
    breakpoint()
    we