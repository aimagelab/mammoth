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
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import logging
import os
import sys
import time
import importlib
import socket
import datetime
import uuid
import argparse
import torch

torch.set_num_threads(2)

# if file is launched inside the `utils` folder
if os.path.dirname(__file__) == 'utils':
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, mammoth_path)

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset

from utils import in_notebook, setup_logging

setup_logging()

if __name__ == '__main__':
    from utils import globals # noqa: F401
    
    logging.info(f"Running Mammoth! on {socket.gethostname()}. ")
    logging.debug("If you see this message more than once, you are probably importing something wrong!")

    from utils.conf import warn_once
    try:
        if os.getenv('MAMMOTH_TEST', '0') == '0':
            from dotenv import load_dotenv
            load_dotenv()
        else:
            warn_once("Running in test mode. Ignoring .env file.")
    except ImportError:
        warn_once("Warning: python-dotenv not installed. Ignoring .env file.")


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def merge_namespace(args: argparse.Namespace, ckpt_args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge the command line arguments with the checkpoint arguments.
    If a key is present in both, the value from the command line arguments takes precedence.
    """
    if ckpt_args is None:
        return args

    # convert both args to dictionaries
    args_dict = vars(args)
    ckpt_args_dict = vars(ckpt_args)

    # merge the dictionaries
    out_dict = ckpt_args_dict.copy()  # start with the checkpoint args
    for key, value in args_dict.items():
        if value is None and key in ckpt_args_dict:
            # if the value is None, use the value from the checkpoint
            out_dict[key] = ckpt_args_dict[key]
        else:
            # otherwise, use the value from the command line arguments
            out_dict[key] = value
        
    # convert back to Namespace
    return argparse.Namespace(**out_dict)

def check_args(args, dataset=None):
    """
    Just a (non complete) stream of asserts to ensure the validity of the arguments.
    """
    assert args.label_perc_by_class == 1 or args.label_perc == 1, "Cannot use both `label_perc_by_task` and `label_perc_by_class`"

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert not args.enable_other_metrics, "Joint training does not support other metrics"
        assert not args.eval_future, "Joint training does not support future evaluation (what is the future?)"

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    if args.savecheck:
        assert not args.inference_only, "Should not save checkpoint in inference only mode"

    assert (args.noise_rate >= 0.) and (args.noise_rate <= 1.), "Noise rate must be in [0, 1]"

    if dataset is not None:
        from datasets.utils.gcl_dataset import GCLDataset, ContinualDataset

        if isinstance(dataset, GCLDataset):
            assert args.n_epochs == 1, "GCLDataset is not compatible with multiple epochs"
            assert args.enable_other_metrics == 0, "GCLDataset is not compatible with other metrics (i.e., forward/backward transfer and forgetting)"
            assert args.eval_future == 0, "GCLDataset is not compatible with future evaluation"
            assert args.noise_rate == 0, "GCLDataset is not compatible with automatic noise injection"

        assert issubclass(dataset.__class__, ContinualDataset) or issubclass(dataset.__class__, GCLDataset), "Dataset must be an instance of `ContinualDataset` or `GCLDataset`"

        if dataset.SETTING == 'biased-class-il':
            assert not args.eval_future, 'Evaluation of future tasks is not supported for biased-class-il.'
            assert not args.enable_other_metrics, 'Other metrics are not supported for biased-class-il.'

        # check if dataset is single-label multi-class (i.e, the `get_loss` returns the cross-entropy)
        if 'cross_entropy' not in str(dataset.get_loss()) and 'CrossEntropy' not in str(dataset.get_loss()):
            if args.noise_rate != 1:
                logging.warning('Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.')


def load_configs(parser: argparse.ArgumentParser, cmd: Optional[List[str]] = None) -> dict:
    from models import get_model_class
    from models.utils import load_model_config

    from datasets import get_dataset_class
    from datasets.utils import get_default_args_for_dataset, load_dataset_config
    from utils.args import fix_model_parser_backwards_compatibility, get_single_arg_value

    cmd = cmd or sys.argv[1:]  # get the command line arguments, if not provided use sys.argv

    args = parser.parse_known_args(cmd)[0]

    # load the model configuration
    # - get the model parser and fix the get_parser function for backwards compatibility
    model_group_parser = parser.add_argument_group('Model-specific arguments')
    model_parser = get_model_class(args).get_parser(model_group_parser)
    parser = fix_model_parser_backwards_compatibility(model_group_parser, model_parser)
    is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])
    buffer_size = None
    if is_rehearsal:  # get buffer size
        buffer_size = get_single_arg_value(parser, 'buffer_size')
        assert buffer_size is not None, "Buffer size not found in the arguments. Please specify it with --buffer_size."
        try:
            buffer_size = int(buffer_size)  # try convert to int, check if it is a valid number
        except ValueError:
            raise ValueError(f'--buffer_size must be an integer but found {buffer_size}')

    # - get the defaults that were set with `set_defaults` in the parser
    base_config = parser._defaults.copy()

    # - get the configuration file for the model
    model_config = load_model_config(args, buffer_size=buffer_size)

    # update the dataset class with the configuration
    dataset_class = get_dataset_class(args)

    # load the dataset configuration. If the model specified a dataset config, use it. Otherwise, use the dataset configuration
    base_dataset_config = get_default_args_for_dataset(args.dataset)
    if 'dataset_config' in model_config:  # if the dataset specified a dataset config, use it
        cnf_file_dataset_config = load_dataset_config(model_config['dataset_config'], args.dataset)
    else:
        cnf_file_dataset_config = load_dataset_config(args.dataset_config, args.dataset)

    dataset_config = {**base_dataset_config, **cnf_file_dataset_config}
    dataset_config = dataset_class.set_default_from_config(dataset_config, parser)  # the updated configuration file is cleaned from the dataset-specific arguments

    # - merge the dataset and model configurations, with the model configuration taking precedence
    config = {**dataset_config, **base_config, **model_config}

    return config


def add_help(parser):
    """
    Add the help argument to the parser
    """
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')


def parse_args(
    cmd: Optional[List[str]] = None, return_parser_only: bool = False, verbose: bool = False
) -> Union[argparse.Namespace, argparse.ArgumentParser]:
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.
    This function initializes the argument parser, adds the necessary arguments, and parses the command line arguments.
    It also loads the configuration files for the model and dataset, sets defaults, and performs some checks on the arguments.
    If `return_parser_only` is True, it returns the parser instead of the parsed arguments.

    Args:
        cmd (Optional[List[str]]): Command line arguments to parse. If None, uses `sys.argv[1:]`.
        return_parser_only (bool): If True, returns the parser instead of the parsed arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from utils import create_if_not_exists
    from utils.conf import warn_once
    from utils.args import (
        add_initial_args,
        add_management_args,
        add_experiment_args,
        add_configuration_args,
        clean_dynamic_args,
        check_multiple_defined_arg_during_string_parse,
        add_dynamic_parsable_args,
        update_cli_defaults,
        get_single_arg_value,
        pretty_format_args,
        set_defaults_args
    )

    cmd = cmd or sys.argv[1:]  # get the command line arguments, if not provided use sys.argv

    check_multiple_defined_arg_during_string_parse()

    parser = argparse.ArgumentParser(
        description="Mammoth - A benchmark Continual Learning framework for Pytorch",
        allow_abbrev=False,
        add_help=False,
    )

    # 1) if loadcheck is provided, load the args from the checkpoint as defaults
    parser.add_argument('--loadcheck', type=str, default=None, help='Path of the checkpoint to load (.pt file for the specific task)')
    args = parser.parse_known_args(cmd)[0]

    ckpt_args = None
    if args.loadcheck is not None:
        from utils.checkpoints import mammoth_load_checkpoint

        # load the checkpoint and set the defaults  
        ckpt_args = mammoth_load_checkpoint(args.loadcheck, return_only_args=True)
        ckpt_args.device = None  # remove the device from the checkpoint args, as it will be set later

    # 2) add arguments that include model, dataset, and backbone. These define the rest of the arguments.
    #   the backbone is optional as may be set by the dataset or the model. The dataset and model are required.
    add_initial_args(parser, strict=ckpt_args is None)
    try:
        # redirect stderr and stdout if we want only the parser
        if return_parser_only:
            import contextlib
            with contextlib.redirect_stderr(None), contextlib.redirect_stdout(None):
                args = parser.parse_known_args(cmd)[0]
        else:
            # parse the arguments
            args = parser.parse_known_args(cmd)[0]
    except SystemExit as e:
        if return_parser_only:
            return parser
        raise e
    
    # - merge args with the defaults from the checkpoint (if it was loaded)
    args = merge_namespace(args, ckpt_args)
    set_defaults_args(parser, **vars(args))  # set the defaults in the parser

    if args.backbone is None and verbose:
        logging.warning('No backbone specified. Using default backbone (set by the dataset).')

    # 3) load the configuration arguments for the dataset and model
    add_configuration_args(parser, args)

    config = load_configs(parser, cmd)

    add_help(parser)

    # 4) add the remaining arguments

    # - get the chosen backbone. The CLI argument takes precedence over the configuration file.
    backbone = args.backbone
    if backbone is None:
        if 'backbone' in config and config['backbone'] is not None:
            backbone = config['backbone']
        else:
            backbone = get_single_arg_value(parser, 'backbone')
    assert backbone is not None, "Backbone not found in the arguments. Please specify it with --backbone or in the model or dataset configuration file."

    # - add the dynamic arguments defined by the chosen dataset and model
    add_dynamic_parsable_args(parser, args.dataset, backbone)

    # - add the main Mammoth arguments
    add_management_args(parser)
    add_experiment_args(parser)

    # - merge and set the defaults for the arguments
    args = merge_namespace(args, ckpt_args)
    set_defaults_args(parser, **vars(args))  # set the defaults in the parser

    # 5) Once all arguments are in the parser, we can set the defaults using the loaded configuration
    update_cli_defaults(parser, config)

    # force call type on all default values to fix values (https://docs.python.org/3/library/argparse.html#type)
    for action in parser._actions:
        if action.default is not None and action.type is not None:
            if action.nargs is None or action.nargs == 0:
                action.default = action.type(action.default)
            else:
                if not isinstance(action.default, (list, tuple)) or (action.type is not list and action.type is not tuple):
                    action.default = [action.type(v) for v in action.default]

    # 6) parse the arguments
    if args.load_best_args:
        from utils.best_args import best_args
        
        if verbose:
            warn_once("The `load_best_args` option is untested and not up to date.")

        is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])  # check if model has a buffer

        args = parser.parse_args(cmd)
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if is_rehearsal:
            best = best[args.buffer_size]
        else:
            best = best[-1]

        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        if return_parser_only:
            return parser
        args = parser.parse_args(to_parse)
        if (
            args.model == "joint" and args.dataset == "mnist-360"
        ):  # TODO: remove this hack -> check if dataset is instance of GCLDataset
            args.model = "joint_gcl"
    else:
        if return_parser_only:
            return parser
        args = parser.parse_args(cmd)

    # 7) clean dynamically loaded args
    args = clean_dynamic_args(args)

    # - merge args with the defaults from the checkpoint (if it was loaded)
    args = merge_namespace(args, ckpt_args)

    # 8) final checks and updates to the arguments
    if args.lr_scheduler is not None and verbose:
        logging.info('`lr_scheduler` set to {}, overrides default from dataset.'.format(args.lr_scheduler))

    if args.seed is not None:
        from utils.conf import set_random_seed

        set_random_seed(args.seed)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # Add the current git commit hash to the arguments if available
    try:
        import git
        repo = git.Repo(path=mammoth_path)
        args.conf_git_hash = repo.head.object.hexsha
    except Exception:
        if verbose:
            logging.error("Could not retrieve git hash.")
        args.conf_git_hash = None

    if args.savecheck:
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        uid = args.conf_jobnum.split('-')[0]
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.dataset_config}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}_{uid}"
        if verbose:
            logging.info(f"Saving checkpoint into: {args.ckpt_name}")

    check_args(args)

    if verbose:
        if args.validation is not None:
            logging.info(f"Using {args.validation}% of the training set as validation set.")
            logging.info(f"Validation will be computed with mode `{args.validation_mode}`.")

        # legacy print of the args, to make automatic parsing easier
        logging.debug(args)

        logging.info('\n' + pretty_format_args(args, parser))

    if in_notebook():
        logging.info("Running in a notebook environment. Forcefully setting num_workers=0 to prevent issues with multiprocessing.")
        args.num_workers = 0

    return args


def extend_args(args, dataset):
    """
    Extend the command-line arguments with the default values from the dataset and the model.
    """
    from datasets import ContinualDataset
    dataset: ContinualDataset = dataset  # noqa, used for type hinting

    if hasattr(args, 'num_classes') and args.num_classes is None:
        args.num_classes = dataset.N_CLASSES


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

    if args.debug_mode:
        logging.warning('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        # set logging level to debug
        args.nowand = 1

    if args.wandb_entity is None:
        args.wandb_entity = os.getenv('WANDB_ENTITY', None)
    if args.wandb_project is None:
        args.wandb_project = os.getenv('WANDB_PROJECT', None)

    if args.wandb_entity is None or args.wandb_project is None:
        logging.info('`wandb_entity` and `wandb_project` not set. Disabling wandb.')
        args.nowand = 1
    else:
        logging.info(f'Logging to wandb: {args.wandb_entity}/{args.wandb_project}')
        args.nowand = 0


def initialize(
    args=None,
) -> Tuple["ContinualModel", "ContinualDataset", argparse.Namespace]:
    from utils.conf import base_path, get_device, get_checkpoint_path
    from models import get_model
    from datasets import get_dataset
    from models.utils.future_model import FutureModel
    from backbone import get_backbone

    lecun_fix()
    if args is None:
        args = parse_args(verbose=True)

    device = get_device(avail_devices=args.device)
    args.device = device

    # set base path
    base_path(args.base_path)

    # set checkpoint path
    get_checkpoint_path(args.checkpoint_path)

    if args.code_optimization != 0:
        torch.set_float32_matmul_precision('high' if args.code_optimization == 1 else 'medium')
        logging.info(f"Code_optimization is set to {args.code_optimization}")
        logging.info(f"Using {torch.get_float32_matmul_precision()} precision for matmul.")

        if args.code_optimization == 2:
            if not torch.cuda.is_bf16_supported():
                raise NotImplementedError('BF16 is not supported on this machine.')

    dataset = get_dataset(args)

    extend_args(args, dataset)

    check_args(args, dataset=dataset)

    backbone = get_backbone(args)
    logging.info(f"Using backbone: {args.backbone}")

    if args.code_optimization == 3:
        # check if the model is compatible with torch.compile
        # from https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        if torch.cuda.get_device_capability()[0] >= 7 and os.name != 'nt':
            logging.warning(
                "\n╔═══════════════════════════ IMPORTANT ══════════════════════════╗\n"
                "║                                                                ║\n"
                "║  Model being compiled with `torch.compile`!                    ║\n"
                "║                                                                ║\n"
                "║  • Your code may break if you modify the model structure       ║\n"
                "║    after the first run                                         ║\n"
                "║                                                                ║\n"
                "║  • This includes adding classifiers for new tasks,             ║\n"
                "║    changing the backbone, etc.                                 ║\n"
                "║                                                                ║\n"
                "║  • Some models MODIFY the backbone during initialization       ║\n"
                "║    Remember to call torch.compile again after such changes     ║\n"
                "║                                                                ║\n"
                "╚════════════════════════════════════════════════════════════════╝"
            )
            backbone = torch.compile(backbone)
        else:
            if torch.cuda.get_device_capability()[0] < 7:
                raise NotImplementedError('torch.compile is not supported on this machine.')
            else:
                raise Exception("torch.compile is not supported on Windows. Check https://github.com/pytorch/pytorch/issues/90768 for updates.")

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform(), dataset=dataset)
    assert isinstance(model, FutureModel) or not args.eval_future, "Model does not support future_forward."

    if args.distributed == 'dp':
        from utils.distributed import make_dp

        if args.batch_size < torch.cuda.device_count():
            raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass

    return model, dataset, args

def main(args=None):
    """
    Main function to run the Mammoth framework.
    It loads the model and dataset, checks the arguments, and starts the training process.
    """
    from utils.training import train

    model, dataset, args = initialize(args)

    train(model, dataset, args)


if __name__ == '__main__':
    main()
