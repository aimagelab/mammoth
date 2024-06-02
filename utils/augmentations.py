"""
This module contains various image augmentation functions and classes.
"""

# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision import transforms
from utils.kornia_utils import KorniaAugNoGrad


def apply_transform(x: torch.Tensor, transform) -> torch.Tensor:
    """Applies a transform to a batch of images.

    If the transforms is a KorniaAugNoGrad, it is applied directly to the batch.
    Otherwise, it is applied to each image in the batch.

    Args:
        x: a batch of images.
        transform: the transform to apply.

    Returns:
        The transformed batch of images.
    """

    if isinstance(transform, KorniaAugNoGrad):
        if isinstance(x, PIL.Image.Image):
            x = torch.as_tensor(np.array(x, copy=True)).permute((2, 0, 1))
        return transform(x)
    else:
        return torch.stack([transform(xi) for xi in x.cpu()], dim=0).to(x.device)


def rand_bbox(size, lam):
    """
    Generate a random bounding box given the size of the image and a lambda value.

    Args:
        size (tuple): The size of the image in the format (batch_size, channels, height, width).
        lam (float): The lambda value used to calculate the size of the bounding box.

    Returns:
        bbx1 (int): The x-coordinate of the top-left corner of the bounding box.
        bby1 (int): The y-coordinate of the top-left corner of the bounding box.
        bbx2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        bby2 (int): The y-coordinate of the bottom-right corner of the bounding box.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    """
    Generate a cutmix sample given a batch of images and labels.

    Args:
        x (torch.Tensor): The batch of images.
        y (torch.Tensor): The batch of labels.
        alpha (float): The alpha value used to calculate the size of the bounding box.
        cutmix_prob (float): The probability of applying cutmix.

    Returns:
        x (torch.Tensor): The mixed batch of images.
        y_a (torch.Tensor): The batch of labels for the first image.
        y_b (torch.Tensor): The batch of labels for the second image.
        lam (float): The lambda value used to calculate the size of the bounding box.

    Raises:
        AssertionError: If the input tensor `x` does not have 4 dimensions.
    """

    if np.random.rand() > cutmix_prob:
        return x, y, y, 1.

    assert (alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def normalize(x, mean, std):
    """
    Normalize the input tensor `x` of images using the provided mean and standard deviation.

    Args:
        x (torch.Tensor): Input tensor to be normalized.
        mean (list or tuple): Mean values for each channel.
        std (list or tuple): Standard deviation values for each channel.

    Returns:
        torch.Tensor: Normalized tensor.

    Raises:
        AssertionError: If the input tensor `x` does not have 4 dimensions.
    """
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)


def random_flip(x):
    """
    Randomly flips the input tensor along the last dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Flipped tensor with the same shape as the input tensor.
    """
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_grayscale(x, prob=0.2):
    """
    Apply random grayscale transformation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        prob (float): Probability of applying the grayscale transformation.

    Returns:
        torch.Tensor: Transformed tensor with random grayscale applied.
    """
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


def random_crop(x, padding):
    """
    Randomly crops the input tensor.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).
        padding (int): The padding size for the crop.

    Returns:
        torch.Tensor: The cropped tensor with shape (batch_size, channels, height, width).
    """
    assert len(x.shape) == 4
    crop_x = torch.randint(-padding, padding, size=(x.shape[0],))
    crop_y = torch.randint(-padding, padding, size=(x.shape[0],))

    crop_x_start, crop_y_start = crop_x + padding, crop_y + padding
    crop_x_end, crop_y_end = crop_x_start + x.shape[-1], crop_y_start + x.shape[-2]

    oboe = F.pad(x, (padding, padding, padding, padding))
    mask_x = torch.arange(x.shape[-1] + padding * 2).repeat(x.shape[0], x.shape[-1] + padding * 2, 1)
    mask_y = mask_x.transpose(1, 2)
    mask_x = ((mask_x >= crop_x_start.unsqueeze(1).unsqueeze(2)) & (mask_x < crop_x_end.unsqueeze(1).unsqueeze(2)))
    mask_y = ((mask_y >= crop_y_start.unsqueeze(1).unsqueeze(2)) & (mask_y < crop_y_end.unsqueeze(1).unsqueeze(2)))
    return oboe[mask_x.unsqueeze(1).repeat(1, x.shape[1], 1, 1) * mask_y.unsqueeze(1).repeat(1, x.shape[1], 1, 1)].reshape(x.shape[0], 3, x.shape[2], x.shape[3])


class soft_aug():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return normalize(
            random_flip(
                random_crop(x, 4)
            ),
            self.mean, self.std)


class strong_aug():
    """
    A class representing a strong data augmentation pipeline (used in X-DER).

    Args:
        size (int): The size of the output image.
        mean (float): The mean value for normalization.
        std (float): The standard deviation value for normalization.
    """

    def __init__(self, size, mean, std):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.ToTensor()
        ])
        self.mean = mean
        self.std = std

    def __call__(self, x):
        flip = random_flip(x)
        return normalize(random_grayscale(
            torch.stack(
                [self.transform(a) for a in flip]
            )), self.mean, self.std)


class DoubleTransform(object):
    """
    This class applies a given transformation to the first image and leaves the second input unchanged.

    Args:
        tf: The transformation to be applied.
    """

    def __init__(self, tf):
        self.transform = tf

    @torch.no_grad()
    def __call__(self, img, other_img):
        """
        Applies the transformation to the first image and leaves the second unchanged.

        Args:
            img: The first image.
            other_img: The second image.

        Returns:
            The transformed first image and the unchanged second image.

        """
        return self.transform(img), other_img


class CustomRandomHorizontalFlip(object):
    """
    Custom augmentation class for performing random horizontal flips on a pair of stackable images and other associated tensors (e.g. attention maps).

    Args:
        p (float): Probability of applying the horizontal flip. Defaults to 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    @torch.no_grad()
    def __call__(self, img, other_img=None):
        """
        Apply random horizontal flip to the input image and to the other associated inputs.

        This transform applies the same random horizontal flip to the input image and to the other associated inputs.

        Args:
            img (PIL.Image.Image): The input image.
            other_img (list[PIL.Image.Image]): List of other images to be flipped.

        Returns:
            PIL.Image.Image: The flipped input image.
            list[PIL.Image.Image]: List of flipped other images.
        """
        if np.random.rand() < self.p:
            return TF.hflip(img), [TF.hflip(x.unsqueeze(0)).squeeze(0) for x in other_img]
        return img, other_img


class CustomRandomCrop(object):
    """
    Custom augmentation class for performing random crop on a pair of stackable images and other associated tensors (e.g. attention maps).

    Args:
        size (int or tuple): Desired output size for the crop. If size is an int, a square crop of size (size, size) is returned.
        padding (int or tuple, optional): Optional padding on each border of the image. Default is 0.
        resize (bool, optional): Whether to resize the other_img maps. Default is False.
        min_resize_index (int, optional): The minimum index of other_img maps to resize. Default is None.

    Returns:
        tuple: A tuple containing the cropped image and a list of cropped other_img maps.
    """

    def __init__(self, size, padding=0, resize=False, min_resize_index=None):
        self.size = size
        self.padding = padding
        self.resize = resize
        self.min_resize_index = min_resize_index
        self.transform = transforms.RandomCrop(size, padding)

    @torch.no_grad()
    def __call__(self, img, other_img=None):
        """
        Apply random crop to the input image and to the other associated inputs.

        This transform applies the same transform to the input image and to the other associated inputs.

        Args:
            img (PIL.Image.Image): The input image.
            other_img (list[PIL.Image.Image]): List of other images to be cropped.

        Returns:
            PIL.Image.Image: The cropped input image.
            list[PIL.Image.Image]: List of cropped other images.
        """
        img = TF.pad(img, self.padding)
        i, j, h, w = self.transform.get_params(img, self.size)

        maps = []
        for idx, map in enumerate(other_img):
            m = map.unsqueeze(0)
            orig_size = m.shape[-2:]
            if self.resize:
                if self.min_resize_index is None or idx <= self.min_resize_index:
                    m = TF.resize(m, (int(orig_size[0] * 2), int(orig_size[1] * 2)), interpolation=transforms.InterpolationMode.NEAREST)

            rate = (self.size[0] // m.shape[-1])
            _i, _j, _h, _w = i // rate, j // rate, h // rate, w // rate
            m = TF.pad(m, self.padding // rate)
            m = TF.crop(m, _i, _j, _h, _w)

            if self.resize:
                if self.min_resize_index is None or idx <= self.min_resize_index:
                    m = TF.resize(m, orig_size, interpolation=transforms.InterpolationMode.NEAREST)

            maps.append(m.squeeze(0))
        return TF.crop(img, i, j, h, w), maps


class DoubleCompose(object):
    """
    Composes multiple transformations to be applied on a pair of stackable images and other associated tensors (e.g. attention maps).

    Args:
        transforms (list): List of transformations to be applied. The transformations should accept two inputs (img, other_img) and return two outputs (img, other_img). For example, :class:`CustomRandomCrop` and :class:`CustomRandomHorizontalFlip`.

    Methods:
        __iter__(): Returns an iterator for the transformations.
        __getitem__(i): Returns the transformation at index i.
        __len__(): Returns the number of transformations.
        __call__(img, other_img): Applies the composed transformations on the input images.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, i):
        return self.transforms[i]

    def __len__(self):
        return len(self.transforms)

    @torch.no_grad()
    def __call__(self, img, other_img):
        other_img = [o.clone() for o in other_img]
        img = img.clone() if isinstance(img, torch.Tensor) else img.copy()
        for t in self.transforms:
            img, other_img = t(img, other_img)
        return img, other_img
