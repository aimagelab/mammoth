"""
Modified version of torchvision.datasets.CelebA to include bias labels and preprocessed splits.
"""

import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from torchvision.datasets import VisionDataset

CSV = namedtuple("CSV", ["header", "index", "data"])


class BiasCelebA(VisionDataset):

    base_folder = "celeba"
    # There currently does not appear to be an easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        version: int = 1,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.target_type = "attr"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        mode = "biased_celeba{}.csv".format(version)
        bias_celeba_path = os.path.join(root, self.base_folder, mode)

        if not os.path.exists(bias_celeba_path):
            if version == 1:
                from datasets.bias_celeba_utils.create_celeba_split1 import process_split
            elif version == 2:
                from datasets.bias_celeba_utils.create_celeba_split2 import process_split
            else:
                raise ValueError("Version {} not supported".format(version))

            process_split(os.path.join(root, self.base_folder))

        splits = pd.read_csv(bias_celeba_path)["partition"]

        # all columns without the image_id, task_number, male and partition
        attr = pd.read_csv(bias_celeba_path).drop(["image_id", "Task_Number", "partition", "Male", "Aligned_With_Bias"], axis=1)
        attr = torch.tensor(attr.values).squeeze()

        task_number = pd.read_csv(bias_celeba_path)["Task_Number"]
        task_number = torch.tensor(task_number.values).squeeze()

        bias_label = pd.read_csv(bias_celeba_path)["Aligned_With_Bias"]
        bias_label = torch.tensor(bias_label.values).squeeze()

        # get split indices
        if split_ is not None:
            mask = splits == split_
        else:
            mask = torch.ones(len(attr), dtype=bool)

        image_ids = pd.read_csv(bias_celeba_path)["image_id"]

        if split_ is not None:
            mask = splits == split_
        else:
            mask = torch.ones(len(attr), dtype=bool)

        self.data = image_ids[mask]
        self.image_folder = os.path.join(self.root, self.base_folder, "img_align_celeba")
        # self.targets = np.array([attr[index, :] for index in range(len(self.data))])

        self.targets = attr[mask]
        self.task_number = task_number[mask]
        self.bias_label = bias_label[mask]

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is not installed. Run `pip install gdown`.")

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)
        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_id, target = self.data.iloc[index], self.targets[index]
        X = Image.open(os.path.join(self.image_folder, img_id))

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self) -> int:
        return len(self.targets)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
