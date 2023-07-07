# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np
import os

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    if torch.cuda.is_available():
        gpu_stats = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
        gpu_stats = [int(x) for x in gpu_stats.split('\n') if x]
        return torch.device('cuda:' + str(np.argmin(gpu_stats)))
    else:
        return torch.device('cpu')

def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'

def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'


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
    except:
        print('Could not set cuda seed.')
        pass
