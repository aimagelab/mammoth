"""
Utility functions for models.
"""

from argparse import ArgumentParser, Namespace
import os
import sys

import yaml

from utils import smart_joint
from utils.conf import warn_once


def load_model_config(args: Namespace, buffer_size: int = None) -> dict:
    """
    Loads the configuration file for the model.

    Args:
        args: the arguments which contains the hyperparameters
        buffer_size: if a method has a buffer, knowing the buffer_size is required to load the best configuration

    Returns:
        dict: the configuration of the model
    """
    filepath = smart_joint('models', 'config', args.model + '.yaml')
    if hasattr(args, 'model_config') and args.model_config:
        assert args.model_config in ['best', 'default']
        if not os.path.exists(filepath):
            if args.model_config == 'best':
                raise FileNotFoundError(f'Model configuration file {args.model_config} not found in {filepath}')
            else:
                warn_once(f'Trying to load default configuration for model {args.model} but no configuration file found in {filepath}.')
                return {}
    else:
        if not os.path.exists(filepath):
            return {}

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    if 'default' not in config:
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
            return {**default_config, **config[args.dataset][buffer_size]}
        return {**default_config, **config[args.dataset]}
