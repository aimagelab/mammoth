"""
Utility functions for models.
"""

from argparse import ArgumentParser, Namespace
import os
import sys

import yaml

from utils import smart_joint
from utils.conf import warn_once


def find_file_ignore_underscores(file: str, basepath=None) -> str:
    """
    Returns the file name by ignoring the underscores and dashes.

    Args:
        file: the file name
        basepath: the base path to search the file

    Returns:
        str: the name of the file that more closely resembles the original file name
    """

    basepath = basepath if basepath is not None else os.getcwd()
    files = os.listdir(basepath)
    file = file.replace('_', '').replace('-', '')
    for f in files:
        if f.replace('_', '').replace('-', '') == file:
            return smart_joint(basepath, f)
    return None


def load_model_config(args: Namespace, buffer_size: int = None) -> dict:
    """
    Loads the configuration file for the model.

    Args:
        args: the arguments which contains the hyperparameters
        buffer_size: if a method has a buffer, knowing the buffer_size is required to load the best configuration

    Returns:
        dict: the configuration of the model
    """
    filepath = find_file_ignore_underscores(args.model + '.yaml', smart_joint('models', 'config'))
    if hasattr(args, 'model_config') and args.model_config:
        assert args.model_config in ['best', 'default']
        if filepath is None or not os.path.exists(filepath):
            if args.model_config == 'best':
                raise FileNotFoundError(f'Model configuration file {args.model_config} not found in {filepath}')
            else:
                warn_once(f'Trying to load default configuration for model {args.model} but no configuration file found in {filepath}.')
                return {}
    else:
        if filepath is None or not os.path.exists(filepath):
            return {}

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    if config is None or 'default' not in config:
        warn_once(f'No default configuration found in {filepath}.')
        default_config = {}
    else:
        default_config = config['default']

    if args.model_config == 'default':
        return default_config
    else:
        assert args.dataset in config, f'No best configuration found in {filepath} for dataset {args.dataset}.'
        if buffer_size is not None:
            assert buffer_size in config[args.dataset], f'No best configuration found in {filepath} for buffer size {buffer_size}.'

            buffer_config = config[args.dataset][buffer_size]  # get arguments for the buffer size only

            other_dataset_config = config[args.dataset]  # get arguments for the dataset only
            del other_dataset_config[buffer_size]

            return {**default_config, **other_dataset_config, **buffer_config}  # merge all arguments, with the buffer size overwriting the dataset
        return {**default_config, **config[args.dataset]}
