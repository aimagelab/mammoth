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
import logging
import numpy  # noqa
import os
import sys
import time
import importlib
import socket
import datetime
import uuid
from argparse import ArgumentParser
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from utils import create_if_not_exists, custom_str_underscore
from utils.conf import warn_once

if __name__ == '__main__':
    try:
        if os.getenv('MAMMOTH_TEST', '0') == '0':
            from dotenv import load_dotenv
            load_dotenv()
        else:
            warn_once("Running in test mode. Ignoring .env file.")
    except ImportError:
        warn_once("Warning: python-dotenv not installed. Ignoring .env file.")

from utils.args import add_management_args, add_experiment_args, fix_argparse_default_priority
from utils.conf import base_path, get_device
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
    from models import get_all_models, get_model_class
    from datasets import get_dataset_names, get_dataset
    # from datasets.utils import update_default_args

    parser = ArgumentParser(description='mammoth', allow_abbrev=False, add_help=False)
    parser.add_argument('--model', type=custom_str_underscore, help='Model name.', choices=list(get_all_models().keys()))
    parser.add_argument('--load_best_args', action='store_true',
                        help='(deprecated) Loads the best arguments for each method, dataset and memory buffer. '
                        'NOTE: This option is deprecated and not up to date.')

    args = parser.parse_known_args()[0]
    models_dict = get_all_models()
    if args.model is None:
        print('No model specified. Please specify a model with --model to see all other options.')
        print('Available models are: {}'.format(list(models_dict.keys())))
        sys.exit(1)

    mod = importlib.import_module('models.' + models_dict[args.model])

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=get_dataset_names(),
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

        parser = get_model_class(args).get_parser()
        add_management_args(parser)
        add_experiment_args(parser)
        fix_argparse_default_priority(parser)
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        parser = get_model_class(args).get_parser()
        add_management_args(parser)
        add_experiment_args(parser)
        fix_argparse_default_priority(parser)
        args = parser.parse_args()

    get_dataset(args).update_default_args()

    args.model = models_dict[args.model]

    if args.lr_scheduler is not None:
        logging.info('`lr_scheduler` set to {}, overrides default from dataset.'.format(args.lr_scheduler))

    if args.seed is not None:
        set_random_seed(args.seed)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # Add the current git commit hash to the arguments if available
    try:
        import git
        repo = git.Repo(path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.conf_git_hash = repo.head.object.hexsha
    except Exception:
        logging.error("Could not retrieve git hash.")
        args.conf_git_hash = None

    if args.savecheck:
        assert args.inference_only == 0, "Should not save checkpoint in inference only mode"
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        uid = args.conf_jobnum.split('-')[0]
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}_{uid}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert args.enable_other_metrics == 0, "Joint training does not support other metrics"

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    if args.validation is not None:
        logging.info(f"Using {args.validation}% of the training set as validation set.")
        logging.info(f"Validation will be computed with mode `{args.validation_mode}`.")

    return args


def main(args=None):
    from models import get_model
    from datasets import ContinualDataset, get_dataset
    from utils.training import train
    from models.utils.future_model import FutureModel

    lecun_fix()
    if args is None:
        args = parse_args()

    device = get_device(avail_devices=args.device)
    args.device = device

    # set base path
    base_path(args.base_path)

    if args.code_optimization != 0:
        torch.set_float32_matmul_precision('high' if args.code_optimization == 1 else 'medium')
        logging.info("Code_optimization is set to", args.code_optimization)
        logging.info(f"Using {torch.get_float32_matmul_precision()} precision for matmul.")

        if args.code_optimization == 2:
            if not torch.cuda.is_bf16_supported():
                raise NotImplementedError('BF16 is not supported on this machine.')

    dataset = get_dataset(args)

    if args.fitting_mode == 'epochs' and args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    elif args.fitting_mode == 'iters' and args.n_iters is None and isinstance(dataset, ContinualDataset):
        args.n_iters = dataset.get_iters()

    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    if args.validation:
        if args.validation_mode == 'current':
            assert dataset.SETTING in ['class-il', 'task-il'], "`current` validation modes is only supported for class-il and task-il settings (requires a task division)."

    backbone = dataset.get_backbone()
    if args.code_optimization == 3:
        # check if the model is compatible with torch.compile
        # from https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        if torch.cuda.get_device_capability()[0] >= 7 and os.name != 'nt':
            print("================ Compiling model with torch.compile ================")
            logging.warning("`torch.compile` may break your code if you change the model after the first run!")
            print("This includes adding classifiers for new tasks, changing the backbone, etc.")
            print("ALSO: some models CHANGE the backbone during initialization. Remember to call `torch.compile` again after that.")
            print("====================================================================")
            backbone = torch.compile(backbone)
        else:
            if torch.cuda.get_device_capability()[0] < 7:
                raise NotImplementedError('torch.compile is not supported on this machine.')
            else:
                raise Exception(f"torch.compile is not supported on Windows. Check https://github.com/pytorch/pytorch/issues/90768 for updates.")

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    # model = torch.compile(model)
    assert isinstance(model, FutureModel) or not args.eval_future, "Model does not support future_forward."

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

    if args.wandb_entity is None:
        args.wandb_entity = os.getenv('WANDB_ENTITY', None)
    if args.wandb_project is None:
        args.wandb_project = os.getenv('WANDB_PROJECT', None)

    if args.wandb_entity is None or args.wandb_project is None:
        logging.warning('`wandb_entity` and `wandb_project` not set. Disabling wandb.')
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
