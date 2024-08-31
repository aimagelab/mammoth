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

_logger = logging.getLogger('dataset/utils')

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


def update_default_args_with_dataset_defaults(parser: ArgumentParser, args: Namespace, dataset_config: dict, strict=True):
    """
    Updates the default arguments with the ones specified in the dataset class and the configuration file.
    Default arguments are defined in the DEFAULT_ARGS dictionary and set by the 'set_default_from_args' decorator.

    .. note::

        The command line arguments have the highest priority. Then the default values defined in the dataset class.
        Finally, the values defined in the configuration file are used.

    Args:
        parser (ArgumentParser): the instance to the argument parser to get metadata about the arguments
        args (Namespace): the arguments to update
        dataset_config (dict): the configuration of the dataset, loaded from the .yaml configuration file
        strict (bool): if True, raises a warning if the argument is not present in the arguments
    """

    if args.dataset not in DEFAULT_ARGS:  # no default args for this dataset
        return

    action_keys = {a.dest: a for a in parser._actions}

    for k, v in DEFAULT_ARGS[args.dataset].items():
        if not hasattr(args, k):
            if strict:
                raise ValueError(f'Argument {k} set by the `set_default_from_args` decorator is not present in the arguments.')
            else:
                continue

        cmd_v = getattr(args, k)
        if cmd_v is None or (action_keys[k].nargs is not None and isinstance(cmd_v, (list, tuple)) and cmd_v == []):  # no command line argument is provided, try the default
            v = dataset_config.get(k, v)  # use the dataset configuration if available, else use the default set by `set_default_from_args`
            v = _clean_value(v, action_keys[k])

            if cmd_v != v:
                _logger.info('{} set to {} instead of {}.'.format(k, getattr(args, k), v))
            setattr(args, k, v)


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
