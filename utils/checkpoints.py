
from argparse import Namespace
from collections.abc import Iterable
import copy
import logging
import random
import string
from typing import Dict, Union
import numpy as np
import torch
import os

from tqdm import tqdm
import urllib.request as request

from utils import smart_joint


def to_parsable_obj(r: Union[Dict, Namespace, list, torch.Tensor, np.ndarray]) -> Union[Dict, list, str, int, float, bool]:
    """
    Convert a non-builtin object to a parsable (and loadable with `weights_only=True`) object.
    Looking at you, Namespace.
    """

    if isinstance(r, Namespace):
        return to_parsable_obj(vars(r))
    if isinstance(r, list):
        return [to_parsable_obj(x) for x in r]
    if isinstance(r, dict):
        return {k: to_parsable_obj(v) for k, v in r.items()}
    else:
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy().tolist()
        elif isinstance(r, np.ndarray):
            r = r.tolist()
        if not isinstance(r, str) and isinstance(r, Iterable) and len(r) > 1:
            return [to_parsable_obj(x) for x in r]
        # check if type of r is builtin
        if isinstance(r, (int, float, str, bool)):
            try:
                r = r.item()  # could be numpy scalar
            except BaseException:
                return r
        return None


def _load_mammoth_model(dict_keys, model: torch.nn.Module, args):
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
    return model


def _load_net(dict_keys, model: torch.nn.Module, args, ignore_classifier=True):
    """
    Load a model from a checkpoint. Handles DataParallel and DistributedDataParallel checkpoints.
    If ignore_classifier is True, the classifier weights are not loaded.
    """
    for k in list(dict_keys):
        if args.distributed != 'dp':
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        elif 'module' not in k:
            if 'net' in k:
                dict_keys[k.replace('net.', 'net.module.')] = dict_keys.pop(k)
            else:
                dict_keys[f'module.{k}'] = dict_keys.pop(k)

    if not ignore_classifier:
        cl_weights = [dict_keys[k] for k in list(dict_keys.keys()) if 'classifier' in k]
        if len(cl_weights) > 0:
            cl_size = cl_weights[-1].shape[0]
            model.net.classifier = torch.nn.Linear(
                model.net.classifier.in_features, cl_size).to(model.device)
    else:
        for k in list(dict_keys):
            if 'classifier' in k:
                dict_keys.pop(k)

    for k in list(dict_keys):
        if '_features' in dict_keys:
            dict_keys.pop(k)
    for k in list(dict_keys):
        if 'net' in k:
            dict_keys[k[4:]] = dict_keys.pop(k)
    for k in list(dict_keys):
        if 'wrappee.' in k:
            dict_keys[k.replace('wrappee.', '')] = dict_keys.pop(k)

    try:
        model.net.load_state_dict(dict_keys)
    except BaseException:
        _, unm = model.net.load_state_dict(dict_keys, strict=False)
        unm = [k for k in unm if '_features' not in k and 'linear' not in k]
        if ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None or len(unm) == 0, f"Missing keys: {unm}"

    model.net.to(model.device)
    return model


def _get_random_filename(length=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _download_from_raw_url(url: str, root: str, filename: str = None) -> str:
    os.makedirs(root, exist_ok=True)
    filename = _get_random_filename() + '.pth' if filename is None else filename

    download_target = smart_joint(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    with request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def mammoth_load_checkpoint(args, model: torch.nn.Module, ignore_classifier=False) -> torch.nn.Module:
    """
    Loads the keys from the given checkpoint.
    - Handles DataParallel and DistributedDataParallel checkpoints.
    - Handles checkpoints from previous versions of the code.
    - Handles head initialization for LUCIR.

    Args:
        args: the model with the checkpoint loaded.
        model: the model to be loaded.
        ignore_classifier: whether to ignore the classifier weights.

    Returns:
        the model with the checkpoint loaded.
    """
    # check if checkpoint is a URL
    if args.loadcheck.startswith('http'):
        if 'sharepoint' in args.loadcheck:
            try:
                from onedrivedownloader import download
            except ImportError:
                raise ImportError('OneDriveDownloader is required to download from Sharepoint. Please install it with "pip install onedrivedownloader"')

            logging.info('Downloading checkpoint using OneDriveDownloader...')
            args.loadcheck = download(args.loadcheck, filename='checkpoints/', unzip=True, unzip_path='checkpoints/', clean=True)
        elif 'drive.google.com' in args.loadcheck:
            try:
                from google_drive_downloader import GoogleDriveDownloader as gdd
            except ImportError:
                raise ImportError('GoogleDriveDownloader is required to download from Google Drive. Please install it with "pip install googledrivedownloader"')

            logging.info('Downloading checkpoint using GoogleDriveDownloader...')
            # get random filename
            filename = _get_random_filename()
            gdd.download_file_from_google_drive(file_id=args.loadcheck.split('/')[-2],
                                                dest_path=f'checkpoints/{filename}', unzip=True)
            args.loadcheck = f'checkpoints/{filename}'
        elif args.loadcheck.startswith('https://huggingface.co/'):
            logging.info('Downloading checkpoints from HuggingFace...')
            filename = args.loadcheck.split('/')[-1].split('?')[0]
            args.loadcheck = _download_from_raw_url(args.loadcheck, 'checkpoints/', filename=filename)
        else:
            logging.warning('Attempting to download raw checkpoint. Make sure to check the URL.')
            args.loadcheck = _download_from_raw_url(args.loadcheck, 'checkpoints/')

        logging.info(f'Checkpoint downloaded to {args.loadcheck}')
    else:
        if not os.path.exists(args.loadcheck):
            raise ValueError('The given checkpoint does not exist.')

    saved_obj = torch.load(args.loadcheck, map_location=torch.device("cpu"), weights_only=True)

    if 'args' in saved_obj and 'model' in saved_obj:
        saved_obj['args'] = Namespace(**saved_obj['args'])  # convert back to Namespace
        _check_loaded_args(args, saved_obj['args'])
        # Mammoth checkpoint
        model = _load_mammoth_model(saved_obj['model'], model, args)
        if 'buffer' in saved_obj:
            loading_model = saved_obj['args'].model
            if args.model != loading_model:
                logging.warning(f'The loaded model was trained with a different model: {loading_model}')
            model.load_buffer(saved_obj['buffer'])

        return model, saved_obj['results']
    else:
        # Model only checkpoint
        model = _load_net(saved_obj, model, args, ignore_classifier=ignore_classifier)

        return model, None


def save_mammoth_checkpoint(task: int, end_task: int, args: Namespace, model: torch.nn.Module, results=None,
                            optimizer_st: Dict[str, torch.Tensor] = None,
                            scheduler_st: Dict[str, torch.Tensor] = None):
    """
    Save a checkpoint for the model for the given task.
    Handles saving as a single file (will require `weights_only=False)` or separate weights (can be loaded safely with `weights_only=True`).
    """
    if args.savecheck == 'task':
        checkpoint_name = f'checkpoints/{args.ckpt_name}_joint' if args.joint else f'checkpoints/{args.ckpt_name}_{task}'
    elif args.savecheck == 'last':
        if task == end_task - 1:
            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint' if args.joint else f'checkpoints/{args.ckpt_name}_last'
        else:
            return
    else:
        raise ValueError(f'Invalid savecheck mode: {args.savecheck}')

    if args.save_checkpoint_mode == 'old_pickle':
        save_obj = {
            'model': model.state_dict(),
            'args': args,
            'results': results,
            'optimizer': optimizer_st,
            'scheduler': scheduler_st,
        }
        if 'buffer_size' in model.args:
            save_obj['buffer'] = copy.deepcopy(model.buffer).to('cpu')
    elif args.save_checkpoint_mode == 'safe':  # TODO CHECK
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer_st,
            'scheduler': scheduler_st,
            'args': to_parsable_obj(vars(args)),  # avoid Namespace and other non-builtin types
            'results': to_parsable_obj(results),  # avoid numpy, torch, and non-builtin types
        }
        if 'buffer_size' in model.args:
            save_obj['buffer'] = model.buffer.serialize()

    torch.save(save_obj, checkpoint_name + '.pt')
    logging.info(f"Checkpoint for task {task} saved at {checkpoint_name}")


def _check_loaded_args(args, loaded_args):
    pruned_original_args = to_parsable_obj(vars(args))

    def _check_arg(arg, loaded_arg):
        if isinstance(arg, (list, tuple)):
            return any([a != la for a, la in zip(arg, loaded_arg)])
        elif isinstance(arg, dict):
            return any([k not in loaded_arg or _check_arg(v, loaded_arg[k]) for k, v in arg.items()])
        elif isinstance(arg, (torch.Tensor, np.ndarray)):
            return (arg != loaded_arg).any()
        return arg != loaded_arg

    ignored_args = ['loadcheck', 'start_from', 'stop_after', 'conf_jobnum', 'conf_host', 'conf_timestamp', 'distributed', 'examples_log', 'examples_full_log',
                    'intensive_savecheck', 'job_number', 'conf_git_commit', 'loss_log', 'tensorboard', 'seed', 'savecheck', 'notes', 'non_verbose', 'autorelaunch',
                    'force_compat', 'conf_external_path', 'ckpt_name']
    mismatched_args = [x for x in pruned_original_args if x not in ignored_args and (
        x not in vars(loaded_args) or _check_arg(pruned_original_args[x], getattr(loaded_args, x)))]

    if len(mismatched_args):
        if 'force_compat' not in vars(args) or args.force_compat:
            logging.warning("The following arguments do not match between loaded and current model:")
            logging.warning(mismatched_args)
        else:
            raise ValueError('The loaded model was trained with different arguments: {}'.format(mismatched_args))
