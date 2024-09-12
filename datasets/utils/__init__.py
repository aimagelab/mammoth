"""
This package contains utility functions used by all datasets, including the base dataset class (ContinualDataset).
"""

from argparse import ArgumentParser, Namespace
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


def _clean_value(value, argparse_action):
    """
    - Converts the value to a list if it is defined with 'nargs' in the argparse action. Can split values by space or comma.
    - Converts the values 'None', 'True', and 'False' to their respective python values.
    """
    if argparse_action.nargs is not None and not isinstance(value, (list, tuple)):
        if isinstance(value, str):
            try:
                value = eval(value)  # check if the value is parsable e.g. '[1, 2, 3]'
            except BaseException:
                if ' ' in value:  # split by space
                    value = [v.strip() for v in value.split()]
                elif ',' in value:  # split by comma
                    value = [v.strip() for v in value.split(',')]
                else:
                    value = [value.strip()]
            if argparse_action.nargs == '?' and len(value) == 1:
                value = value[0]

    def _to_python_value(v):
        if not isinstance(v, str):
            return v
        if v == 'None':
            return None
        if v == 'True':
            return True
        if v == 'False':
            return False
        return v

    if isinstance(value, (list, tuple)):
        return [_to_python_value(v) for v in value]
    return _to_python_value(value)


def get_default_args_for_dataset(dataset_name: str) -> dict:
    """
    Get the default arguments defined by `set_default_from_args` for the given dataset.

    Args:
        dataset_name (str): the name of the dataset

    Returns:
        dict: the default arguments for the dataset
    """
    return DEFAULT_ARGS.get(dataset_name, {})


def load_dataset_config(dataset_config: str, dataset: str) -> dict:
    """
    Loads the configuration file for the dataset.

    Args:
        dataset_config (str): the name of the configuration file
        dataset (str): the name of the dataset

    Returns:
        dict: the configuration of the dataset
    """
    if dataset_config:
        assert isinstance(dataset_config, str), f"Invalid dataset configuration file: {dataset_config}. Specify a string."
        filepath = smart_joint('datasets', 'configs', dataset, dataset_config + '.yaml')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Dataset configuration file {dataset_config} not found in {filepath}')
    else:
        filepath = smart_joint('datasets', 'configs', dataset, 'default.yaml')
        if not os.path.exists(filepath):
            warn_once(f'Default configuration file not found for dataset {dataset}. '
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
