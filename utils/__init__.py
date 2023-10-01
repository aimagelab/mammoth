# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple
import torch


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def mammoth_load_checkpoint(args, model: torch.nn.Module) -> torch.nn.Module:
    """
    Loads the keys from the given checkpoint.
    - Handles DataParallel and DistributedDataParallel checkpoints.
    - Handles checkpoints from previous versions of the code.
    - Handles head initialization for LUCIR.
    :param args: the model with the checkpoint loaded.
    """   
    if not os.path.exists(args.loadcheck):
        raise ValueError('The given checkpoint does not exist.')
    
    saved_obj = torch.load(args.loadcheck, map_location=torch.device("cpu"))
    _check_loaded_args(args, saved_obj['args'])

    dict_keys = saved_obj['model']
    for k in list(dict_keys):
        if args.distributed != 'dp':
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        elif 'module' not in k:
            dict_keys[k.replace('net.', 'net.module.')] = dict_keys.pop(k)

    for k in list(dict_keys):
        if '_features' in dict_keys:
            dict_keys.pop(k)

    if 'lucir' in args.model.lower():
        model.register_buffer('classes_so_far', torch.zeros_like(
            dict_keys['classes_so_far']).to('cpu'))
    
    model.load_state_dict(dict_keys)
    model.net.to(model.device)

    if 'buffer' in saved_obj:
        loading_model = saved_obj['args'].model
        if args.model != loading_model:
            print(f'WARNING: The loaded model was trained with a different model: {loading_model}')
        model.load_buffer(saved_obj['buffer'])

    return model, saved_obj['results']

def _check_loaded_args(args, loaded_args):
    ignored_args = ['loadcheck', 'start_from', 'stop_after', 'conf_jobnum', 'conf_host', 'conf_timestamp', 'distributed', 'examples_log', 'examples_full_log',
                    'intensive_savecheck', 'job_number', 'conf_git_commit', 'loss_log', 'tensorboard', 'seed', 'savecheck', 'notes', 'non_verbose', 'autorelaunch', 'force_compat', 'conf_external_path']
    mismatched_args = [x for x in vars(args) if x not in ignored_args and (
        x not in vars(loaded_args) or getattr(args, x) != getattr(loaded_args, x))]
    
    if len(mismatched_args):
        if 'force_compat' not in vars(args) or args.force_compat:
            print(
                "WARNING: The following arguments do not match between loaded and current model:")
            print(mismatched_args)
        else:
            raise ValueError(
                'The loaded model was trained with different arguments: {}'.format(mismatched_args))
