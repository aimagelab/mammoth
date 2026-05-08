import logging
import os
import shutil
import torch
import pickle
from typing import List, Dict
from typing import Tuple, Union
from tqdm.auto import tqdm
import json
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


def load_and_preprocess_cars196(class_names: List[str], dataset: Dict[str, List[torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor, dict], dict]:
    """
    Loads data from deeplake and preprocesses it to be stored locally.

    Args:
        class_names: list of class names
        datasett: dataset to be pre-processed

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict] | dict: If names_only is False, returns a tuple of data, targets, and class_idx_to_name
    """
    class_idx_to_name = {i: class_names[i] for i in range(len(class_names))}

    # Pre-process dataset
    data = []
    targets = []
    for x in tqdm(dataset, desc=f'Pre-processing dataset'):
        img = x['images'][0].permute(2, 0, 1)  # load one image at a time
        if len(img) < 3:
            img = img.repeat(3, 1, 1)  # fix rgb
        img = MyCars196.PREPROCESSING_TRANSFORM(img)  # resize
        data.append(img)
        label = x['car_models'][0].item()  # get label
        targets.append(label)

    data = torch.stack(data)  # stack all images
    targets = torch.tensor(targets)

    return data, targets, class_idx_to_name


class MyCars196(Dataset):
    N_CLASSES = 196
    HF_REPO_ID = 'aimagelab-ta/cars196'
    HF_REPO_TYPE = 'dataset'
    HF_REVISION = 'main'
    REQUIRED_FILES = (
        'train_images.pt',
        'train_labels.pt',
        'test_images.pt',
        'test_labels.pt',
    )
    CLASS_NAME_FILES = ('class_names.json', 'class_names.pkl')

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    PREPROCESSING_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
    ])

    def __init__(self, root, train=True, transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.ToTensor()

        self.ensure_dataset_files(root)

        train_str = 'train' if train else 'test'
        if os.path.exists(f'{root}/{train_str}_images.pt'):
            logging.info(f"Loading pre-processed {train_str} dataset...")
            self.data = torch.load(f'{root}/{train_str}_images.pt', weights_only=True)
            self.targets = torch.load(f'{root}/{train_str}_labels.pt', weights_only=True)
            if os.path.exists(f'{root}/class_names.json'):
                with open(f'{root}/class_names.json', 'rt') as f:
                    self.class_names = json.load(f)
            elif os.path.exists(base_path() + f'cars196/class_names.pkl'):
                with open(base_path() + f'cars196/class_names.pkl', 'rb') as f:
                    self.class_names = pickle.load(f)
        else:
            raise FileNotFoundError(
                f'Could not find preprocessed split files in `{root}` after attempting download from '
                f'`{MyCars196.HF_REPO_ID}`.'
            )

        self.class_names = MyCars196.get_class_names()

    @classmethod
    def _download_file_from_hf(cls, root: str, filename: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                'huggingface_hub is required to download CARS196 automatically. '
                'Install it with `pip install huggingface_hub`.'
            ) from e

        downloaded_path = hf_hub_download(
            repo_id=cls.HF_REPO_ID,
            filename=filename,
            repo_type=cls.HF_REPO_TYPE,
            revision=cls.HF_REVISION,
        )

        target_path = os.path.join(root, filename)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(downloaded_path, target_path)

    @classmethod
    def ensure_dataset_files(cls, root: str) -> None:
        os.makedirs(root, exist_ok=True)

        missing_required = [
            filename for filename in cls.REQUIRED_FILES
            if not os.path.exists(os.path.join(root, filename))
        ]
        has_class_names = any(
            os.path.exists(os.path.join(root, filename))
            for filename in cls.CLASS_NAME_FILES
        )

        if not missing_required and has_class_names:
            return

        if missing_required:
            logging.info(
                'Missing CARS196 files in `%s`. Downloading from `hf://%s@%s`.',
                root,
                cls.HF_REPO_ID,
                cls.HF_REVISION,
            )
            for filename in missing_required:
                cls._download_file_from_hf(root, filename)

        if not has_class_names:
            class_name_error = None
            for filename in cls.CLASS_NAME_FILES:
                try:
                    cls._download_file_from_hf(root, filename)
                    has_class_names = True
                    break
                except Exception as e:
                    class_name_error = e
            if not has_class_names:
                raise FileNotFoundError(
                    f'Could not find/download class names for CARS196 in `{root}`. '
                    f'Expected one of: {cls.CLASS_NAME_FILES}.'
                ) from class_name_error

    def load_and_preprocess_dataset(self, root, train_str='train'):
        self.data, self.targets, class_idx_to_name = load_and_preprocess_cars196(self.class_names, self.dataset)

        logging.info(f"Saving pre-processed dataset in {root} ({train_str}_images.pt and {train_str}_labels.py)...")
        if not os.path.exists(root):
            os.makedirs(root)
        torch.save(self.data, f'{root}/{train_str}_images.pt')
        torch.save(self.targets, f'{root}/{train_str}_labels.pt')

        with open(f'{root}/class_names.json', 'wt') as f:
            json.dump(class_idx_to_name, f, indent=4)
        logging.info('Done')

    @staticmethod
    def get_class_names():
        if os.path.exists(base_path() + f'cars196/class_names.json'):
            with open(base_path() + f'cars196/class_names.json', 'rt') as f:
                class_idx_to_name = json.load(f)
            class_names = list(class_idx_to_name.values())
        elif os.path.exists(base_path() + f'cars196/class_names.pkl'):
            with open(base_path() + f'cars196/class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
        else:
            raise ValueError("Class names not found")
        return class_names

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCars196(ContinualDataset):
    """
    Sequential CARS196 Dataset. The images are loaded from deeplake, resized to 224x224, and store locally.
    """

    NAME = 'seq-cars196'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 196
    N_CLASSES_PER_TASK = [20] * 9 + [16]
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])  # no transform for test

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCars196(base_path() + 'cars196', train=True,
                                  transform=self.TRANSFORM)
        test_dataset = MyCars196(base_path() + 'cars196', train=False,
                                 transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_prompt_templates():
        return templates['cars196']

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = MyCars196.get_class_names()
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCars196.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialCars196.MEAN, std=SequentialCars196.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCars196.MEAN, SequentialCars196.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128
