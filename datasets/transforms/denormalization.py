# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class DeNormalize(object):
    def __init__(self, mean, std):
        """
        Initializes a DeNormalize object.

        Args:
            mean (list): List of mean values for each channel.
            std (list): List of standard deviation values for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Applies denormalization to the input tensor.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.

        Returns:
            Tensor: Denormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
