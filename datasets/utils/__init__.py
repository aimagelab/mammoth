"""
This package contains utility functions used by all datasets, including the base dataset class (ContinualDataset).
"""

from argparse import Namespace
import functools
import inspect
import logging
import os
from typing import List

import yaml
from torchvision import transforms

from utils import smart_joint
from utils.conf import warn_once

# Default arguments defined by the datasets
# The structure is {dataset_name: {arg_name: default_value}}
DEFAULT_ARGS = {}


def is_static_call(*args) -> bool:
    """
    Check if the function is called without any arguments.

    Returns:
        bool: True if the function is called without any arguments, False otherwise.
    """
    return len(args) == 0


def set_default_from_args(arg_name: str):
    """
    Decorator to define the default value of an argument of a given dataset.

    Args:
        arg_name (str): The name of the argument to set the default value for.

    Returns:
        function: The decorator to set the default value of the argument.
    """

    global DEFAULT_ARGS
    caller = inspect.currentframe().f_back
    caller_name = caller.f_locals['NAME']
    if caller_name not in DEFAULT_ARGS:
        DEFAULT_ARGS[caller_name] = {}

    def decorator_set_default_from_args(func):
        n_args = len(inspect.signature(func).parameters)
        if arg_name in DEFAULT_ARGS[caller_name]:
            raise ValueError(f"Argument `{arg_name}` already has a default value in `{caller_name}`")
        if n_args == 1:  # has self
            DEFAULT_ARGS[caller_name][arg_name] = func(None)
        else:
            DEFAULT_ARGS[caller_name][arg_name] = func()

        @functools.wraps(func)
        def wrapper(*args):

            if is_static_call(*args):
                # if no arguments are passed, return the function
                return func(None)

            return func(*args)
        return wrapper
    return decorator_set_default_from_args


def update_args_with_dataset_defaults(args: Namespace, strict=True):
    """
    Updates the default arguments with the ones specified in the dataset class.
    Default arguments are defined in the DEFAULT_ARGS dictionary and set by the 'set_default_from_args' decorator.

    Args:
        args (Namespace): the arguments to update
        strict (bool): if True, raises a warning if the argument is not present in the arguments
    """

    if args.dataset not in DEFAULT_ARGS:  # no default args for this dataset
        return

    for k, v in DEFAULT_ARGS[args.dataset].items():
        if not hasattr(args, k):
            if strict:
                raise ValueError(f'Argument {k} set by the `set_default_from_args` decorator is not present in the arguments.')
            else:
                continue

        if getattr(args, k) is None:
            setattr(args, k, v)
        else:
            if getattr(args, k) != v:
                logging.info('{} set to {} instead of {}.'.format(k, getattr(args, k), v))


def load_config(args: Namespace) -> dict:
    """
    Loads the configuration file for the dataset.

    Args:
        args: the arguments which contains the hyperparameters

    Returns:
        dict: the configuration of the dataset
    """
    if hasattr(args, 'dataset_config') and args.dataset_config:
        filepath = smart_joint('datasets', 'configs', args.dataset, args.dataset_config + '.yaml')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Dataset configuration file {args.dataset_config} not found in {filepath}')
    else:
        filepath = smart_joint('datasets', 'configs', args.dataset, 'default.yaml')
        if not os.path.exists(filepath):
            warn_once(f'Default configuration file not found for dataset {args.dataset}. '
                      'Using the defaults specified in the dataset class (if available).')
            return {}

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

        return config


def build_torchvision_transform(transform_specs: List[dict]) -> transforms.Compose:
    """
    Builds the transformation pipeline from the given specifications.

    Args:
        transform_specs (List[dict]): the specifications of the transformations

    Returns:
        transforms.Compose: the transformation pipeline
    """
    if not isinstance(transform_specs, list):
        transform_specs = [transform_specs]

    transform_list = []
    for spec in transform_specs:
        if isinstance(spec, str):
            transform_list.append(getattr(transforms, spec)())
        else:
            assert isinstance(spec, dict), f"Invalid transformation specification: {spec}"
            for k, v in spec.items():
                if isinstance(v, dict):
                    transform_list.append(getattr(transforms, k)(**v))
                else:
                    transform_list.append(getattr(transforms, k)(v))

    return transforms.Compose(transform_list)
