# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class Permutation(object):
    """
    Defines a fixed permutation for a numpy array.
    """
    def __init__(self) -> None:
        """
        Initializes the permutation.
        """
        self.perm = None

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """
        Randomly defines the permutation and applies the transformation.
        :param sample: image to be permuted
        :return: permuted image
        """
        old_shape = sample.shape
        if self.perm is None:
            self.perm = np.random.permutation(len(sample.flatten()))

        return sample.flatten()[self.perm].reshape(old_shape)


class FixedPermutation(object):
    """
    Defines a fixed permutation (given the seed) for a numpy array.
    """
    def __init__(self, seed: int) -> None:
        """
        Defines the seed.
        :param seed: seed of the permutation
        """
        self.perm = None
        self.seed = seed

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """
        Defines the permutation and applies the transformation.
        :param sample: image to be permuted
        :return: permuted image
        """
        old_shape = sample.shape
        if self.perm is None:
            np.random.seed(self.seed)
            self.perm = np.random.permutation(len(sample.flatten()))

        return sample.flatten()[self.perm].reshape(old_shape)
