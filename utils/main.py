"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Sets up the necessary paths and configurations.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# needed (don't change it)
import numpy  # noqa
import time
import importlib
import os
import socket
import sys
import datetime
import uuid
from argparse import ArgumentParser
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from utils import create_if_not_exists
from utils.conf import base_path
from utils.distributed import make_dp
from utils.best_args import best_args
from utils.conf import set_random_seed


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from models import get_all_models
    from datasets import NAMES as DATASET_NAMES, get_dataset_class

    parser = ArgumentParser(description='mammoth', allow_abbrev=False, add_help=False)
    parser.add_argument('--model', type=str, help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    args = parser.parse_known_args()[0]
    if args.model is None:
        print('No model specified. Please specify a model with --model to see all other options.')
        print('Available models are: {}'.format(get_all_models()))
        sys.exit(1)

    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    tmp_dset_class = get_dataset_class(args)
    n_epochs = tmp_dset_class.get_epochs()
    if args.n_epochs is None:
        args.n_epochs = n_epochs
    else:
        if args.n_epochs != n_epochs:
            print('Warning: n_epochs set to {} instead of {}'.format(args.n_epochs, n_epochs), file=sys.stderr)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.savecheck:
        assert args.inference_only == 0, "Should not save checkpoint in inference only mode"
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}"
        args.ckpt_name_replace = f"{extra_ckpt_name}{args.model}_{args.dataset}_{'{}'}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}__{args.n_epochs}_{str(now)}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert args.enable_other_metrics == 0, "Joint training does not support other metrics"

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    return args


def main(args=None) -> None:
    """
    Main function for training a model on a dataset.

    Args:
        args (argparse.Namespace): Optional. Command-line arguments. If not specified, they are parsed from the command line.

    Returns:
        None
    """
    from models import get_model
    from datasets import ContinualDataset, get_dataset
    from utils.training import train

    # Rest of the code...


def main(args=None):
    from models import get_model
    from datasets import ContinualDataset, get_dataset
    from utils.training import train

    lecun_fix()
    if args is None:
        args = parse_args()

    # set base path
    base_path(args.base_path)

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == 'dp':
        if args.batch_size < torch.cuda.device_count():
            raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        args.nowand = 1

    if args.wandb_entity is None or args.wandb_project is None:
        print('Warning: wandb_entity and wandb_project not set. Disabling wandb.')
        args.nowand = 1
    else:
        print('Logging to wandb: {}/{}'.format(args.wandb_entity, args.wandb_project))
        args.nowand = 0

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass

    train(model, dataset, args)


if __name__ == '__main__':
    main()
