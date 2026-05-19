import io
import json
import logging
import os
from pathlib import Path
import zipfile
import requests
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple

from datasets.utils import set_default_from_args
from datasets.utils.hf_download import download_dataset_snapshot_with_patterns
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class MyEuroSat(Dataset):
    HF_REPO_ID = 'aimagelab-ta/eurosat-rgb'
    HF_REVISION = 'main'
    HF_PARQUET_REVISION = 'refs/convert/parquet'
    READY_FILE = 'DONE'

    @staticmethod
    def _download_split_from_gdrive(dest_path: str) -> None:
        try:
            from google_drive_downloader import GoogleDriveDownloader as gdd
        except ImportError as e:
            raise ImportError(
                'google_drive_downloader is required for legacy EuroSat split download. '
                'Install it with `pip install googledrivedownloader==0.4`.'
            ) from e
        gdd.download_file_from_google_drive(
            file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
            dest_path=dest_path,
        )

    @staticmethod
    def _download_legacy(root: str) -> None:
        logging.info('Preparing EuroSat dataset from legacy sources...')
        os.makedirs(root, exist_ok=True)
        r = requests.get('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(root)
        os.system(f'mv {root}/EuroSAT_RGB/* {root}')
        os.system(f'rmdir {root}/EuroSAT_RGB')
        with open(root + '/' + MyEuroSat.READY_FILE, 'w') as f:
            f.write('')

        MyEuroSat._download_split_from_gdrive(dest_path=root + '/split.json')
        logging.info('Done')

    @staticmethod
    def _images_from_parquet(root: str) -> None:
        import pandas as pd
        from tqdm.auto import tqdm

        split_path = os.path.join(root, 'split.json')
        with open(split_path, 'r') as f:
            split_cfg = json.load(f)

        expected = {}
        for split_name in ('train', 'test', 'val'):
            for item in split_cfg[split_name]:
                relpath = item[0]
                expected[os.path.basename(relpath)] = relpath

        parquet_files = sorted(str(p) for p in Path(root).rglob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError('No parquet files found for EuroSAT materialization')

        matched = 0
        written = 0
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file, columns=['image'])
            for image in tqdm(df['image'], total=len(df), desc=f'Materializing {os.path.basename(parquet_file)}', leave=False):
                if not isinstance(image, dict):
                    continue
                image_bytes = image.get('bytes')
                image_name = image.get('path')
                if image_bytes is None or image_name is None:
                    continue

                relpath = expected.get(os.path.basename(image_name))
                if relpath is None:
                    continue
                matched += 1

                out_path = os.path.join(root, relpath)
                if os.path.isfile(out_path):
                    continue

                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(image_bytes)
                written += 1

        if matched == 0:
            raise FileNotFoundError('Parquet images do not match split.json paths for EuroSAT')
        logging.info('Materialized %d EuroSAT images from parquet (%d matched records)', written, matched)

    @staticmethod
    def ensure_dataset_files(root: str) -> None:
        os.makedirs(root, exist_ok=True)
        ready_path = os.path.join(root, MyEuroSat.READY_FILE)
        split_path = os.path.join(root, 'split.json')

        if os.path.isfile(ready_path) and os.path.isfile(split_path):
            return

        try:
            if not os.path.isfile(split_path):
                download_dataset_snapshot_with_patterns(
                    local_dir=root,
                    repo_id=MyEuroSat.HF_REPO_ID,
                    revision=MyEuroSat.HF_REVISION,
                    allow_patterns='split.json',
                )

            if not os.path.isfile(split_path):
                raise FileNotFoundError(f'Missing EuroSAT split file in `{root}`: split.json')

            download_dataset_snapshot_with_patterns(
                repo_id=MyEuroSat.HF_REPO_ID,
                local_dir=root,
                revision=MyEuroSat.HF_PARQUET_REVISION,
                allow_patterns='**/*.parquet',
            )
            MyEuroSat._images_from_parquet(root=root)

            with open(ready_path, 'w') as f:
                f.write('')
        except Exception as e:
            logging.warning('HF download for EuroSat failed, falling back to legacy sources: %s', e)
            MyEuroSat._download_legacy(root)

        if not os.path.isfile(ready_path) or not os.path.isfile(split_path):
            raise FileNotFoundError(
                f'EuroSAT dataset is not marked as ready in `{root}`. '
                f'Missing `{MyEuroSat.READY_FILE}` or `split.json`.'
            )

    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:
        self.root = root
        self.split = split
        assert split in ['train', 'test', 'val'], 'Split must be either train, test or val'
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.ToTensor()

        MyEuroSat.ensure_dataset_files(self.root)

        with open(os.path.join(self.root, 'split.json'), 'r') as f:
            split_cfg = json.load(f)
        split_data = split_cfg[split]
        self.data = np.array([row[0] for row in split_data])
        self.targets = np.array([row[1] for row in split_data]).astype(np.int16)

        self.class_names = self.get_class_names()

    @staticmethod
    def get_class_names():
        MyEuroSat.ensure_dataset_files(base_path() + 'eurosat')
        with open(os.path.join(base_path() + 'eurosat', 'split.json'), 'r') as f:
            train_data = json.load(f)['train']
        class_names = []
        seen = set()
        for _, _, class_name in train_data:
            if class_name not in seen:
                class_names.append(class_name)
                seen.add(class_name)
        return class_names

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
            MyEuroSat(base_path() + 'eurosat', split='train', transform=None)
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
    def get_epochs():
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    @staticmethod
    def get_prompt_templates():
        return templates['eurosat']
