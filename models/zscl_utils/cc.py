import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SubsetRandomSampler,
    get_worker_info,
)

import models.zscl_utils.clip as clip


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        df = pd.read_csv(input_filename, sep=sep)

        self.location = os.path.dirname(input_filename)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.location, str(self.images[idx]))
        images = self.transforms(Image.open(image_path))
        texts = clip.tokenize([str(self.captions[idx])])[0]
        return images, texts


class conceptual_captions(Dataset):
    def __init__(
        self, transforms, location, batch_size, *args, num_workers=0, **kwargs
    ):
        file_name = "Validation_GCC-1.1.0-Validation_output.csv"
        file_path = os.path.join(location, file_name)
        self.template = lambda c: f"a photo of a {c}."
        self.train_dataset = CsvDataset(
            input_filename=file_path,
            transforms=transforms,
            img_key="filepath",
            caption_key="title",
        )
        # breakpoint()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
