import signal
import uuid
import functools
import os
from argparse import Namespace
import copy
import logging
import random
import string
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch

from tqdm.auto import tqdm
import urllib.request as request

from utils import smart_joint, to_parsable_obj, in_notebook
from utils.globals import GLOBALS 
from utils.conf import get_checkpoint_path

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


def _load_net(dict_keys, model: torch.nn.Module, ignore_classifier=True):
    """
    Load a model from a checkpoint. Handles DataParallel and DistributedDataParallel checkpoints.
    If ignore_classifier is True, the classifier weights are not loaded.
    """
    for k in list(dict_keys):
        if k.startswith('module.'): # remove 'module.' prefix if present
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        else: #remove '.module.' if present
            dict_keys[k.replace('.module.', '.')] = dict_keys.pop(k)

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


def mammoth_load_checkpoint(checkpoint_path: str,
                            model: Optional[torch.nn.Module] = None,
                            ignore_classifier=False,
                            args: Optional[Namespace]=None,
                            return_only_args: bool=False) -> Union[Namespace, Tuple[torch.nn.Module, Optional[Dict[str, Union[float, int]]]]]:
    """
    Loads the keys from the given checkpoint.
    - Handles DataParallel and DistributedDataParallel checkpoints.
    - Handles checkpoints from previous versions of the code.
    - Handles head initialization for LUCIR.

    Args:
        checkpoint_path: the path to the checkpoint file or URL.
        model: the model to be loaded. It can be None ONLY with `return_only_args=True`.
        ignore_classifier: whether to ignore the classifier weights.
        args: the current arguments. If provided, it will check if the loaded arguments match the current ones.
        return_only_args: if True, only returns the loaded arguments and not the model.

    Returns:
        the model with the checkpoint loaded.
    """
    assert model is not None or return_only_args, "Model must be provided if return_only_args is False."

    # check if checkpoint is a URL
    if checkpoint_path.startswith('http'):
        if 'sharepoint' in checkpoint_path:
            try:
                from onedrivedownloader import download
            except ImportError:
                raise ImportError('OneDriveDownloader is required to download from Sharepoint. Please install it with "pip install onedrivedownloader"')

            logging.info('Downloading checkpoint using OneDriveDownloader...')
            checkpoint_path = download(checkpoint_path, filename=get_checkpoint_path(),
                                      unzip=True, unzip_path=get_checkpoint_path(), clean=True)
        elif 'drive.google.com' in checkpoint_path:
            try:
                from google_drive_downloader import GoogleDriveDownloader as gdd
            except ImportError:
                raise ImportError('GoogleDriveDownloader is required to download from Google Drive. Please install it with "pip install googledrivedownloader"')

            logging.info('Downloading checkpoint using GoogleDriveDownloader...')
            # get random filename
            filename = _get_random_filename()
            dest = os.path.join(get_checkpoint_path(), filename)
            gdd.download_file_from_google_drive(file_id=checkpoint_path.split('/')[-2],
                                                dest_path=dest, unzip=True)
            checkpoint_path = dest
        elif checkpoint_path.startswith('https://huggingface.co/'):
            logging.info('Downloading checkpoints from HuggingFace...')
            filename = checkpoint_path.split('/')[-1].split('?')[0]
            checkpoint_path = _download_from_raw_url(checkpoint_path, get_checkpoint_path(), filename=filename)
        else:
            logging.warning('Attempting to download raw checkpoint. Make sure to check the URL.')
            checkpoint_path = _download_from_raw_url(checkpoint_path, get_checkpoint_path())

        logging.info(f'Checkpoint downloaded to {checkpoint_path}')
    else:
        if not os.path.exists(checkpoint_path):
            raise ValueError('The given checkpoint does not exist.')

    saved_obj = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)

    if 'args' in saved_obj:
        ckpt_args = Namespace(**saved_obj['args'])  # convert back to Namespace
        if args:
            _check_loaded_args(args, ckpt_args)
        else:
            args = ckpt_args
        if return_only_args:
            return args
        
        if 'model' in saved_obj:
            # Mammoth checkpoint
            model = _load_mammoth_model(saved_obj['model'], model, args)
            if 'buffer' in saved_obj:
                loading_model = saved_obj['args'].model
                if args.model != loading_model:
                    logging.warning(f'The loaded model was trained with a different model: {loading_model}')
                model.load_buffer(saved_obj['buffer'])

            return model, saved_obj['results']
        else:
            raise ValueError("""The checkpoint is not in a valid format.
Expect a checkpoint either with:
- 'args' and 'model' keys (Mammoth checkpoint)
- simple state_dict WITH NO 'args' KEY""")

    else:
        assert not return_only_args, "Cannot return only args when the checkpoint does not contain them."
        # Model only checkpoint
        model = _load_net(saved_obj, model, ignore_classifier=ignore_classifier)

        return model, None


def save_mammoth_checkpoint(task: int, end_task: int, args: Namespace, model: torch.nn.Module, results=None,
                            optimizer_st: Dict[str, torch.Tensor] = None,
                            scheduler_st: Dict[str, torch.Tensor] = None,
                            checkpoint_name: str = None):
    """
    Save a checkpoint for the model for the given task.
    Handles saving as a single file (will require `weights_only=False)` or separate weights (can be loaded safely with `weights_only=True`).
    """
    if checkpoint_name is None:
        if args.savecheck == 'task':
            checkpoint_name = os.path.join(get_checkpoint_path(), f'{args.ckpt_name}_joint') if args.joint else os.path.join(get_checkpoint_path(), f'{args.ckpt_name}_{task}')
        elif args.savecheck == 'last':
            if task == end_task - 1:
                checkpoint_name = os.path.join(get_checkpoint_path(), f'{args.ckpt_name}_joint') if args.joint else os.path.join(get_checkpoint_path(), f'{args.ckpt_name}_last')   
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

def can_save_and_exit(fn: Callable) -> Callable:
    """
    Wraps a function to catch KeyboardInterrupt and SigInt signals. 

    If running in a Jupyter notebook, this will prevent the kernel from crashing
    when the user interrupts the execution of a cell and retain the current state.

    If running in a script, this will:
     - catch the KeyboardInterrupt and exit gracefully
     - catch the SigInt and save a checkpoint before exiting
    This is useful for training scripts where you want to be able to stop the training
    process and save the current state of the model.

    Args:
        fn: the function to be wrapped

    Returns:
        the wrapped function
    """
    wrapped = hasattr(can_save_and_exit, 'wrapped')
    ckpt_path = get_checkpoint_path()
    tmp_filename = str(uuid.uuid4())
    ckpt_path = os.path.join(ckpt_path, 'paused', tmp_filename)
    if not os.path.exists(os.path.dirname(ckpt_path)):
        os.makedirs(os.path.dirname(ckpt_path))

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not wrapped:
            if not in_notebook():
                signal.signal(signal.SIGINT, _get_sigint_handler(fn, ckpt_path))
                signal.signal(signal.SIGTERM, _get_sigint_handler(fn, ckpt_path))
            else:
                def _ignore_sigint(signum, frame):
                    global GLOBALS
                    logging.info("SIGINT received in notebook. Ignoring to prevent kernel crash.")
                    GLOBALS['SHOULD_STOP'] = True  # type: ignore[assignment]
                signal.signal(signal.SIGINT, _ignore_sigint)

        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            pass

    setattr(can_save_and_exit, 'wrapped', True) # avoid re-registering the signal handler

    return wrapped_fn

def _get_sigint_handler(fn: Callable, ckpt_path: str) -> Callable:
    def _handle_sigint_terminal(signum, frame):
        global GLOBALS

        current = frame
        while current:
            if current.f_code == fn.__code__:
                _locals = current.f_locals
                break
            current = current.f_back

        if 'args' not in _locals: # not initialized yet, can safely exit
            logging.info("SIGINT received before initialization. Exiting...")
            GLOBALS['SHOULD_STOP'] = True

        logging.info("SIGINT received. Saving checkpoint and exiting...")
        exp_args = _locals.get('args')
        model = _locals.get('model')
        scheduler = _locals.get('scheduler')
        if exp_args.save_after_interrupt:
            save_mammoth_checkpoint(_locals['cur_task'], _locals['end_task'], exp_args,
                                    model,
                                    results=[_locals['results'], _locals['results_mask_classes'], _locals['logger'].dump()],
                                    optimizer_st=model.opt.state_dict() if hasattr(model, 'opt') else None,
                                    scheduler_st=scheduler.state_dict() if scheduler is not None else None,
                                    checkpoint_name=ckpt_path)
        
        GLOBALS['SHOULD_STOP'] = True
    return _handle_sigint_terminal