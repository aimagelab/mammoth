# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

from datasets.utils.continual_dataset import ContinualDataset


class GCLDataset(ContinualDataset):
    """
    General Continual Learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    N_CLASSES: int
    SIZE: Tuple[int]

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.

        Args:
            args: the arguments which contains the hyperparameters
        """
        self.N_CLASSES_PER_TASK = self.N_CLASSES
        self.N_TASKS = 1
        assert args.n_epochs == 1, 'GCLDataset is not compatible with multiple epochs'
        super().__init__(args)

        if not all((self.NAME, self.SETTING, self.SIZE)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def get_epochs(self):
        """
        A GCLDataset is not compatible with multiple epochs.
        """

        return 1
