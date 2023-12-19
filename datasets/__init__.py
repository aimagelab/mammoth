# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)

import importlib
import inspect
from argparse import Namespace

from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import warn_once


def get_all_datasets():
    """Returns the list of all the available datasets in the datasets folder."""
    return [model.split('.')[0] for model in os.listdir('datasets')
            if not model.find('__') > -1 and 'py' in model]


NAMES = {}
for dataset in get_all_datasets():
    try:
        mod = importlib.import_module('datasets.' + dataset)
        dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
                                and 'ContinualDataset' in str(inspect.getmro(getattr(mod, x))[1:]) and 'GCLDataset' not in str(inspect.getmro(getattr(mod, x)))]
        for d in dataset_classes_name:
            c = getattr(mod, d)
            NAMES[c.NAME] = c

        gcl_dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
        for d in gcl_dataset_classes_name:
            c = getattr(mod, d)
            NAMES[c.NAME] = c
    except Exception as e:
        warn_once(f'Error in dataset {dataset}')
        warn_once(e)
        NAMES[dataset.replace('_', '-')] = e


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    Args:
        args (Namespace): the arguments which contains the hyperparameters

    Returns:
        the continual dataset
    """
    assert args.dataset in NAMES
    return get_dataset_class(args)(args)


def get_dataset_class(args: Namespace) -> ContinualDataset:
    """
    Returns a continual dataset.

    Args:
        args (Namespace): the arguments which contains the hyperparameters

    Returns:
        the continual dataset
    """
    assert args.dataset in NAMES
    if isinstance(NAMES[args.dataset], Exception):
        raise NAMES[args.dataset]
    return NAMES[args.dataset]
