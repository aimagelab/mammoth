"""
Datasets can be included either by registering them using the `register_dataset` decorator or by following the old naming convention:
- A single dataset is defined in a file named `<dataset_name>.py` in the `datasets` folder.
- The dataset class must inherit from `ContinualDataset`.
"""

import os
import sys
from typing import Callable
import importlib
import inspect
from argparse import Namespace

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)

from utils import infer_args_from_signature, register_dynamic_module_fn
from utils.conf import warn_once
from datasets.utils.continual_dataset import ContinualDataset

REGISTERED_DATASETS = dict()  # dictionary containing the registered datasets. Template: {name: {'class': class, 'parsable_args': parsable_args}}


def register_dataset(name: str) -> Callable:
    """
    Decorator to register a ContinualDatasety. The decorator may be used on a class that inherits from `ContinualDataset` or on a function that returns a `ContinualDataset` instance.
    The registered dataset can be accessed using the `get_dataset` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the dataset's class.
    The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the dataset
    """
    if hasattr(get_dataset_names, 'names'):  # reset the cache of the dataset names
        del get_dataset_names.names

    return register_dynamic_module_fn(name, REGISTERED_DATASETS, ContinualDataset)


def get_all_datasets_legacy():
    """
    Returns the list of all the available datasets in the datasets folder that follow the old naming convention.
    """

    return [model.split('.')[0] for model in os.listdir('datasets')
            if not model.find('__') > -1 and 'py' in model]


def get_dataset_names(names_only=False):
    """
    Return the names of the available continual dataset.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        names_only (bool): whether to return only the names of the available datasets

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the named of the available continual datasets
    """

    def _dataset_names():
        names = {}  # key: dataset name, value: {'class': dataset class, 'parsable_args': parsable_args}
        for dataset, dataset_conf in REGISTERED_DATASETS.items():
            names[dataset.replace('_', '-')] = {'class': dataset_conf['class'], 'parsable_args': dataset_conf['parsable_args']}

        base_class_signature = inspect.signature(ContinualDataset.__init__)
        for dataset in get_all_datasets_legacy():  # for the datasets that follow the old naming convention, load the dataset class and check for errors
            if dataset in names:  # dataset registered with the new convention has priority
                continue

            try:
                mod = importlib.import_module('datasets.' + dataset)
                dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
                                        and 'ContinualDataset' in str(inspect.getmro(getattr(mod, x))[1:]) and 'GCLDataset' not in str(inspect.getmro(getattr(mod, x)))]
                for d in dataset_classes_name:
                    c = getattr(mod, d)
                    signature = inspect.signature(c.__init__)
                    parsable_args = infer_args_from_signature(signature, excluded_signature=base_class_signature)

                    names[c.NAME.replace('_', '-')] = {'class': c, 'parsable_args': parsable_args}

                gcl_dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
                for d in gcl_dataset_classes_name:
                    c = getattr(mod, d)
                    signature = inspect.signature(c.__init__)
                    parsable_args = infer_args_from_signature(signature, excluded_signature=base_class_signature)

                    names[c.NAME.replace('_', '-')] = {'class': c, 'parsable_args': parsable_args}

            except Exception as e:  # if an error is detected, raise the appropriate error message
                warn_once(f'Error in dataset {dataset}')
                warn_once(e)
                names[dataset.replace('_', '-')] = e
        return names

    if not hasattr(get_dataset_names, 'names'):
        setattr(get_dataset_names, 'names', _dataset_names())
    names = getattr(get_dataset_names, 'names')
    if names_only:
        return list(names.keys())
    return names


def get_dataset_config_names(dataset: str):
    """
    Return the names of the available continual dataset configurations.

    The configurations can be used to create a dataset with specific hyperparameters and can be
    specified using the `--dataset_config` attribute.

    The configurations are stored in the `datasets/configs/<dataset>` folder.
    """

    def _dataset_config_names(dataset):
        names = []
        if os.path.exists(f'datasets/configs/{dataset}'):
            names = [dset_config.split('.yaml')[0] for dset_config in os.listdir(f'datasets/configs/{dataset}')
                     if dset_config.endswith('.yaml') and not dset_config.startswith('__')]
        return names

    if not hasattr(get_dataset_config_names, 'names'):
        setattr(get_dataset_config_names, 'names', {})
    if dataset not in get_dataset_config_names.names:
        get_dataset_config_names.names[dataset] = _dataset_config_names(dataset)
    return get_dataset_config_names.names[dataset]


def get_dataset_class(args: Namespace, return_args=False) -> ContinualDataset:
    """
    Return the class of the selected continual dataset among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--dataset` attribute
        return_args (bool): whether to return the parsable arguments of the dataset

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the continual dataset class
    """
    names = get_dataset_names()
    assert args.dataset in names
    if isinstance(names[args.dataset], Exception):
        raise names[args.dataset]
    if return_args:
        return names[args.dataset]['class'], names[args.dataset]['parsable_args']
    return names[args.dataset]['class']


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the hyperparameters

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the continual dataset instance
    """
    dataset_class, dataset_args = get_dataset_class(args, return_args=True)
    missing_args = [arg for arg in dataset_args.keys() if arg not in vars(args)]
    assert len(missing_args) == 0, "Missing arguments for the dataset: " + ', '.join(missing_args)

    parsed_args = {arg: getattr(args, arg) for arg in dataset_args.keys()}

    return dataset_class(args, **parsed_args)


# import all files in the `datasets` folder to register the datasets
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file != '__init__.py':
        importlib.import_module(f'datasets.{file[:-3]}')
