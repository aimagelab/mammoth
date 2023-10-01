from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from backbone.ResNet50 import resnet50
import torch.nn.functional as F
import numpy as np
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import os
from torchvision.transforms import functional as TF
import torch
from torch import nn

class MyCUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224
    N_CLASSES = 200
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
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
                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                print('Downloading dataset')
                download(ln, filename=os.path.join(root, 'cub_200_2011.zip'), unzip=True, unzip_path=root, clean=True)



        data_file = np.load(os.path.join(root, 'train_data.npz' if train else 'test_data.npz'), allow_pickle=True)

        self.data = data_file['data']
        self.targets = torch.from_numpy(data_file['targets']).long()
        self.classes = data_file['classes']
        self.segs = data_file['segs']
        self._return_segmask=False

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        if self._return_segmask:
            raise "Unsupported segmentation output in training set!"
            
        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)


class CUB200(MyCUB200):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    
    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, 'logits') else [img, target]

        if ret_segmask or self._return_segmask:
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode='L')
            seg = transforms.ToTensor()(transforms.CenterCrop((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg>0).int())

        return ret_tuple

class SequentialCUB200(ContinualDataset):
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TRANSFORM = transforms.Compose([ 
         transforms.Resize(MyCUB200.IMG_SIZE),
         transforms.RandomCrop(MyCUB200.IMG_SIZE, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = MyCUB200.TEST_TRANSFORM
    
    def get_data_loaders(self, test_only=False):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((MyCUB200.IMG_SIZE,MyCUB200.IMG_SIZE)), transforms.ToTensor(), self.get_normalization_transform()])
        
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                 download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = CUB200(base_path() + 'CUB200', train=False,
                                  download=True, transform=test_transform)

        class_order = None
        train, test = store_masked_loaders(
            train_dataset, test_dataset, self, class_order)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        num_classes = SequentialCUB200.N_CLASSES_PER_TASK * SequentialCUB200.N_TASKS
        return resnet50(num_classes)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            SequentialCUB200.MEAN, SequentialCUB200.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
        return transform
