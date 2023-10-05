# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np


def get_device() -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device():
        # get least used gpu by used memory
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                gpu_memory.append(torch.cuda.memory_allocated(i))
            device = torch.device(f'cuda:{np.argmin(gpu_memory)}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
        except BaseException:
            print("WARNING: Something went wrong with MPS. Using CPU.")
            return torch.device("cpu")

    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()

    return get_device.device


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'


def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return '/tmp/mammoth_datasets/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        print('Could not set cuda seed.')
        pass
