# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
           / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)

def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x

def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299,0.587,0.114]]).unsqueeze(2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x

def random_crop(x, padding):
    assert len(x.shape) == 4
    crop_x = torch.randint(-padding, padding, size=(x.shape[0],))
    crop_y = torch.randint(-padding, padding, size=(x.shape[0],))

    crop_x_start, crop_y_start = crop_x + padding, crop_y + padding
    crop_x_end, crop_y_end = crop_x_start + x.shape[-1], crop_y_start + x.shape[-2]

    oboe = F.pad(x, (padding, padding, padding, padding))
    mask_x = torch.arange(x.shape[-1] + padding * 2).repeat(x.shape[0], x.shape[-1] + padding * 2, 1)
    mask_y = mask_x.transpose(1,2)
    mask_x = ((mask_x >= crop_x_start.unsqueeze(1).unsqueeze(2)) & (mask_x < crop_x_end.unsqueeze(1).unsqueeze(2)))
    mask_y = ((mask_y >= crop_y_start.unsqueeze(1).unsqueeze(2)) & (mask_y < crop_y_end.unsqueeze(1).unsqueeze(2)))
    return oboe[mask_x.unsqueeze(1).repeat(1,x.shape[1],1,1) * mask_y.unsqueeze(1).repeat(1,x.shape[1],1,1)].reshape(x.shape[0], 3, x.shape[2], x.shape[3])

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
