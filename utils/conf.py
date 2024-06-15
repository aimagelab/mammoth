"""
This module contains utility functions for configuration settings.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import torch
import numpy as np


def warn_once(*msg):
    """
    Prints a warning message only once.

    Args:
        msg: the message to be printed
    """
    msg = ' '.join([str(m) for m in msg])
    if not hasattr(warn_once, 'warned'):
        warn_once.warned = set()
    if msg not in warn_once.warned:
        warn_once.warned.add(msg)
        print(msg, file=sys.stderr)


def get_alloc_memory_all_devices() -> list[int]:
    """
    Returns the memory allocated on all the available devices.
    """
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.memory_allocated(i))
    if all(memory == 0 for memory in gpu_memory):
        print("WARNING: some weird GPU memory issue."
              "Using trick from https://discuss.pytorch.org/t/torch-cuda-memory-allocated-returns-0-if-pytorch-no-cuda-memory-caching-1/188796")
        for i in range(torch.cuda.device_count()):
            torch.zeros(1).to(i)
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            gpu_memory[i] = total_memory - free_memory
    return gpu_memory


def get_device() -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device():
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_memory = get_alloc_memory_all_devices()
            device = torch.device(f'cuda:{np.argmin(gpu_memory)}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("WARNING: MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            print("WARNING: Something went wrong with MPS. Using CPU.")

        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()
        print(f'Using device {get_device.device}')

    return get_device.device


def base_path(override=None) -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.

    Args:
        override: the path to override the default one. Once set, it is stored and used for all the next calls.

    Returns:
        the base path (default: `./data/`)
    """
    if override is not None:
        if not os.path.exists(override):
            os.makedirs(override)
        if not override.endswith('/'):
            override += '/'
        setattr(base_path, 'path', override)

    if not hasattr(base_path, 'path'):
        setattr(base_path, 'path', './data/')
    return getattr(base_path, 'path')


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.

    Args:
        seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        print('Could not set cuda seed.')


def set_random_seed_worker(worker_id) -> None:
    """
    Sets the seeds for a worker of a dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_seeded_dataloader(args, dataset, **dataloader_args) -> torch.utils.data.DataLoader:
    """
    Creates a dataloader object from a dataset, setting the seeds for the workers (if `--seed` is set).

    Args:
        args: the arguments of the program
        dataset: the dataset to be loaded
        dataloader_args: external arguments of the dataloader

    Returns:
        the dataloader object
    """

    n_cpus = 4 if not hasattr(os, 'sched_getaffinity') else len(os.sched_getaffinity(0))
    num_workers = n_cpus if args.num_workers is None else args.num_workers
    dataloader_args['num_workers'] = num_workers if 'num_workers' not in dataloader_args else dataloader_args['num_workers']
    if args.seed is not None:
        worker_generator = torch.Generator()
        worker_generator.manual_seed(args.seed)
    else:
        worker_generator = None
    dataloader_args['generator'] = worker_generator if 'generator' not in dataloader_args else dataloader_args['generator']
    dataloader_args['worker_init_fn'] = set_random_seed_worker if 'worker_init_fn' not in dataloader_args else dataloader_args['worker_init_fn']
    return torch.utils.data.DataLoader(dataset, **dataloader_args)
