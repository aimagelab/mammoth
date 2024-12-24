# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys

if __name__ == '__main__':
    import os
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(mammoth_path)

from argparse import ArgumentParser, Namespace

from backbone import REGISTERED_BACKBONES
from datasets import get_dataset_names, get_dataset_config_names
from models import get_all_models
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type, custom_str_underscore, field_with_aliases


def get_single_arg_value(parser: ArgumentParser, arg_name: str):
    """
    Returns the value of a single argument without explicitly parsing the arguments.

    Args:
        parser: the argument parser
        arg_name: the name of the argument

    Returns:
        str: the value of the argument
    """
    action = [action for action in parser._actions if action.dest == arg_name]
    assert len(action) == 1, f'Argument {arg_name} not found in the parser.'
    action = action[0]

    # if the argument has a default value, return it
    if action.default is not None:
        return action.default

    # otherwise, search for the argument in the sys.argv
    for i, arg in enumerate(sys.argv):
        arg_k = arg.split('=')[0]
        if arg_k in action.option_strings or arg_k == action.dest:
            if len(arg.split('=')) == 2:
                return arg.split('=')[1]
            else:
                return sys.argv[i + 1]
    return None


def update_cli_defaults(parser: ArgumentParser, cnf: dict) -> None:
    """
    Updates the default values of the parser with the values in the configuration dictionary.

    If an argument is defined as `required` in the parser but a default value is provided in the configuration dictionary, the argument is set as not required.

    Args:
        parser: the argument parser
        cnf: the configuration dictionary

    Returns:
        None
    """
    parser.set_defaults(**cnf)

    for action in parser._actions:
        if action.dest == 'help':
            continue
        if action.dest in cnf:
            action.default = cnf[action.dest]
            action.required = False


def fix_model_parser_backwards_compatibility(main_parser: ArgumentParser, model_parser: ArgumentParser = None) -> ArgumentParser:
    """
    Fix the backwards compatibility of the `get_parser` method of the models.

    Args:
        main_parser: the main parser
        model_parser: the parser of the model

    Returns:
        the fixed parser
    """
    if model_parser is None:
        return main_parser

    if main_parser != model_parser:
        for action in model_parser._actions:
            if action.dest == 'help':
                continue
            # add the arguments of the model parser to the main parser
            if not any([action.dest == a.dest for a in main_parser._actions]):
                main_parser._add_action(action)

    # update the defaults of the main parser with the defaults of the model parser
    set_defaults_args = model_parser._defaults

    for action in main_parser._actions:
        if action.dest in set_defaults_args:
            action.default = set_defaults_args[action.dest]
            action.required = False

    return main_parser


def build_parsable_args(parser: ArgumentParser, spec: dict) -> None:
    """
    Builds the argument parser given a specification and extends the given parser.

    The specification dictionary can either be a simple list of key-value argument or follow the format:

    .. code-block:: python

        {
            'name': {
                'type': type,
                'default': default,
                'choices': choices,
                'help': help,
                'required': True/False
            }
        }

    If the specification is a simple list of key-value arguments, the value of the argument is the default value. If the default is set to `inspect.Parameter.empty`, the argument is required. The type of the argument is inferred from the default value (default is `str`).

    Args:
        parser: the argument parser
        spec: the specification dictionary

    Returns:
        the argument parser
    """

    for name, arg_spec in spec.items():
        # check if the argument is already defined in the parser
        if any([action.dest == name for action in parser._actions]):
            logging.warn(f"Argument `{name}` is already defined in the parser. Skipping...")
            continue

        if isinstance(arg_spec, dict):
            arg_type = arg_spec.get('type', str)
            arg_default = arg_spec.get('default', None)
            arg_choices = arg_spec.get('choices', None)
            arg_help = arg_spec.get('help', '')
            arg_required = arg_spec.get('required', False)
        else:
            arg_choices = None
            arg_help = ''
            arg_type = type(arg_spec)
            arg_default = arg_spec
            arg_required = False

        parser.add_argument(f'--{name}', type=arg_type, default=arg_default, choices=arg_choices, help=arg_help, required=arg_required)


def clean_dynamic_args(args: Namespace) -> Namespace:
    """
    Extracts the registered name from the dictionary arguments.
    """
    if isinstance(args.backbone, dict):
        args.backbone = args.backbone['type']
    if isinstance(args.model, dict):
        args.model = args.model['type']
    if isinstance(args.dataset, dict):
        args.dataset = args.dataset['type']
    return args


def add_dynamic_parsable_args(parser: ArgumentParser, dataset: str, backbone: str) -> None:
    """
    Add the additional arguments of the chosen dataset and backbone to the parser.

    Args:
        parser: the parser instance to extend
        dataset: the dataset name
        backbone: the backbone name
    """

    ds_group = parser.add_argument_group('Dataset arguments', 'Arguments used to define the dataset.')
    registered_datasets = get_dataset_names()
    if isinstance(dataset, dict):
        assert 'type' in dataset, "The dataset `type` (i.e., the registered name) must be defined in the dictionary."
        bk_name = dataset['type'].replace('_', '-').lower()
        bk_args = {**registered_datasets[bk_name]['parsable_args'], **dataset['args']}
        dataset = bk_name
    else:
        bk_args = registered_datasets[dataset.replace('_', '-').lower()]['parsable_args']
    build_parsable_args(ds_group, bk_args)

    bk_group = parser.add_argument_group('Backbone arguments', 'Arguments used to define the backbone network.')
    if isinstance(backbone, dict):
        assert 'type' in backbone, "The backbone `type` (i.e., the registered name) must be defined in the dictionary."
        bk_name = backbone['type'].replace('-', '_').lower()
        bk_args = {**REGISTERED_BACKBONES[bk_name]['parsable_args'], **backbone['args']}
        backbone = bk_name
    else:
        bk_args = REGISTERED_BACKBONES[backbone.replace('_', '-').lower()]['parsable_args']
    build_parsable_args(bk_group, bk_args)

    # model dynamic arguments? maybe in the future...


def add_configuration_args(parser: ArgumentParser, args: Namespace) -> None:
    """
    Arguments that need to define the configuration of the dataset and model.
    """

    config_group = parser.add_argument_group('Configuration arguments', 'Arguments used to define the dataset and model configurations.')

    config_group.add_argument('--dataset_config', type=str,
                              choices=get_dataset_config_names(args.dataset),
                              help='The configuration used for this dataset (e.g., number of tasks, transforms, backbone architecture, etc.).'
                              'The available configurations are defined in the `datasets/config/<dataset>` folder.')

    config_group.add_argument('--model_config', type=field_with_aliases({'default': ['base', 'default'], 'best': ['best']}), default='default',
                              help='The configuration used for this model. The available configurations are defined in the `models/config/<model>.yaml` file '
                              'and include a `default` (dataset-agostic) configuration and a `best` configuration (dataset-specific). '
                              'If not provided, the `default` configuration is used.')


def add_initial_args(parser) -> ArgumentParser:
    """
    Returns the initial parser for the arguments.
    """
    parser.add_argument('--dataset', type=custom_str_underscore, required=True,
                        choices=get_dataset_names(names_only=True),
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=custom_str_underscore, required=True,
                        help='Model name.', choices=list(get_all_models().keys()))
    parser.add_argument('--backbone', type=custom_str_underscore, help='Backbone network name.', choices=list(REGISTERED_BACKBONES.keys()))
    parser.add_argument('--load_best_args', action='store_true',
                        help='(deprecated) Loads the best arguments for each method, dataset and memory buffer. '
                        'NOTE: This option is deprecated and not up to date.')

    return parser


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    exp_group = parser.add_argument_group('Experiment arguments', 'Arguments used to define the experiment settings.')

    exp_group.add_argument('--lr', required=True, type=float, help='Learning rate. This should either be set as default by the model '
                           '(with `set_defaults <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.set_defaults>`_),'
                           ' by the dataset (with `set_default_from_args`, see :ref:`module-datasets.utils`), or with `--lr=<value>`.')
    exp_group.add_argument('--batch_size', type=int, help='Batch size.')

    exp_group.add_argument('--label_perc_by_task', '--label_perc', '--lpt', type=float, default=1,
                           dest='label_perc', help='Percentage in (0-1] of labeled examples per task.')
    exp_group.add_argument('--label_perc_by_class', '--lpc', type=float, default=1, dest='label_perc_by_class',
                           help='Percentage in (0-1] of labeled examples per task.')
    exp_group.add_argument('--joint', type=int, choices=(0, 1), default=0, help='Train model on Joint (single task)?')
    exp_group.add_argument('--eval_future', type=binary_to_boolean_type, default=False, help='Evaluate future tasks?')

    validation_group = parser.add_argument_group('Validation and fitting arguments', 'Arguments used to define the validation strategy and the method used to fit the model.')

    validation_group.add_argument('--validation', type=float, help='Percentage of samples FOR EACH CLASS drawn from the training set to build the validation set.')
    validation_group.add_argument('--validation_mode', type=str, choices=['complete', 'current'], default='current',
                                  help='Mode used for validation. Must be used in combination with `validation` argument. Possible values:'
                                  ' - `current`: uses only the current task for validation (default).'
                                  ' - `complete`: uses data from both current and past tasks for validation.')
    validation_group.add_argument('--fitting_mode', type=str, choices=['epochs', 'iters', 'time', 'early_stopping'], default='epochs',
                                  help='Strategy used for fitting the model. Possible values:'
                                  ' - `epochs`: fits the model for a fixed number of epochs (default). NOTE: this option is controlled by the `n_epochs` argument.'
                                  ' - `iters`: fits the model for a fixed number of iterations. NOTE: this option is controlled by the `n_iters` argument.'
                                  ' - `early_stopping`: fits the model until early stopping criteria are met. This option requires a validation set (see `validation` argument).'
                                  '   The early stopping criteria are: if the validation loss does not decrease for `early_stopping_patience` epochs, the training stops.')
    validation_group.add_argument('--early_stopping_patience', type=int, default=5,
                                  help='Number of epochs to wait before stopping the training if the validation loss does not decrease. Used only if `fitting_mode=early_stopping`.')
    validation_group.add_argument('--early_stopping_metric', type=str, default='loss', choices=['loss', 'accuracy'],
                                  help='Metric used for early stopping. Used only if `fitting_mode=early_stopping`.')
    validation_group.add_argument('--early_stopping_freq', type=int, default=1,
                                  help='Frequency of validation evaluation. Used only if `fitting_mode=early_stopping`.')
    validation_group.add_argument('--early_stopping_epsilon', type=float, default=1e-6,
                                  help='Minimum improvement required to consider a new best model. Used only if `fitting_mode=early_stopping`.')
    validation_group.add_argument('--n_epochs', type=int,
                                  help='Number of epochs. Used only if `fitting_mode=epochs`.')
    validation_group.add_argument('--n_iters', type=int,
                                  help='Number of iterations. Used only if `fitting_mode=iters`.')

    opt_group = parser.add_argument_group('Optimizer and learning rate scheduler arguments', 'Arguments used to define the optimizer and the learning rate scheduler.')

    opt_group.add_argument('--optimizer', type=str, default='sgd',
                           choices=ContinualModel.AVAIL_OPTIMS,
                           help='Optimizer.')
    opt_group.add_argument('--optim_wd', type=float, default=0.,
                           help='optimizer weight decay.')
    opt_group.add_argument('--optim_mom', type=float, default=0.,
                           help='optimizer momentum.')
    opt_group.add_argument('--optim_nesterov', type=binary_to_boolean_type, default=0,
                           help='optimizer nesterov momentum.')
    opt_group.add_argument('--drop_last', type=binary_to_boolean_type, default=0,
                           help='Drop the last batch if it is not complete?')
    opt_group.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler.')
    opt_group.add_argument('--scheduler_mode', type=str, choices=['epoch', 'iter'], default='epoch',
                           help='Scheduler mode. Possible values:'
                           ' - `epoch`: the scheduler is called at the end of each epoch.'
                           ' - `iter`: the scheduler is called at the end of each iteration.')
    opt_group.add_argument('--lr_milestones', type=int, default=[], nargs='+',
                           help='Learning rate scheduler milestones (used if `lr_scheduler=multisteplr`).')
    opt_group.add_argument('--sched_multistep_lr_gamma', type=float, default=0.1,
                           help='Learning rate scheduler gamma (used if `lr_scheduler=multisteplr`).')

    noise_group = parser.add_argument_group('Noise arguments', 'Arguments used to define the noisy-label settings.')

    noise_group.add_argument('--noise_type', type=field_with_aliases({
        'symmetric': ['symmetric', 'sym', 'symm'],
        'asymmetric': ['asymmetric', 'asym', 'asymm']
    }), default='symmetric',
        help='Type of noise to apply. The symmetric type is supported by all datasets, while the asymmetric must be supported explicitly by the dataset (see `datasets/utils/label_noise`).')
    noise_group.add_argument('--noise_rate', type=float, default=0,
                             help='Noise rate in [0-1].')
    noise_group.add_argument('--disable_noisy_labels_cache', type=binary_to_boolean_type, default=0,
                             help='Disable caching the noisy label targets? NOTE: if the seed is not set, the noisy labels will be different at each run with this option disabled.')
    noise_group.add_argument('--cache_path_noisy_labels', type=str, default='noisy_labels',
                             help='Path where to save the noisy labels cache. The path is relative to the `base_path`.')


def add_management_args(parser: ArgumentParser) -> None:
    """
    Adds the management arguments.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    mng_group = parser.add_argument_group('Management arguments', 'Generic arguments to manage the experiment reproducibility, logging, debugging, etc.')

    mng_group.add_argument('--seed', type=int, default=None,
                           help='The random seed. If not provided, a random seed will be used.')
    mng_group.add_argument('--permute_classes', type=binary_to_boolean_type, default=0,
                           help='Permute classes before splitting into tasks? This applies the seed before permuting if the `seed` argument is present.')
    mng_group.add_argument('--base_path', type=str, default="./data/",
                           help='The base path where to save datasets, logs, results.')
    mng_group.add_argument('--results_path', type=str, default="results/",
                           help='The path where to save the results. NOTE: this path is relative to `base_path`.')
    mng_group.add_argument('--device', type=str,
                           help='The device (or devices) available to use for training. '
                           'More than one device can be specified by separating them with a comma. '
                           'If not provided, the code will use the least used GPU available (if there are any), otherwise the CPU. '
                           'MPS is supported and is automatically used if no GPU is available and MPS is supported. '
                           'If more than one GPU is available, Mammoth will use the least used one if `--distributed=no`.')
    mng_group.add_argument('--notes', type=str, default=None,
                           help='Helper argument to include notes for this run. Example: distinguish between different versions of a model and allow separation of results')
    mng_group.add_argument('--eval_epochs', type=int, default=None,
                           help='Perform inference on validation every `eval_epochs` epochs. If not provided, the model is evaluated ONLY at the end of each task.')
    mng_group.add_argument('--non_verbose', default=0, type=binary_to_boolean_type, help='Make progress bars non verbose')
    mng_group.add_argument('--disable_log', default=0, type=binary_to_boolean_type, help='Disable logging?')
    mng_group.add_argument('--num_workers', type=int, default=None, help='Number of workers for the dataloaders (default=infer from number of cpus).')
    mng_group.add_argument('--enable_other_metrics', default=0, type=binary_to_boolean_type,
                           help='Enable computing additional metrics: forward and backward transfer.')
    mng_group.add_argument('--debug_mode', type=binary_to_boolean_type, default=0, help='Run only a few training steps per epoch. This also disables logging on wandb.')
    mng_group.add_argument('--inference_only', default=0, type=binary_to_boolean_type,
                           help='Perform inference only for each task (no training).')
    mng_group.add_argument('-O', '--code_optimization', type=int, default=0, choices=[0, 1, 2, 3],
                           help='Optimization level for the code.'
                           '0: no optimization.'
                           '1: Use TF32, if available.'
                           '2: Use BF16, if available.'
                           '3: Use BF16 and `torch.compile`. BEWARE: torch.compile may break your code if you change the model after the first run! Use with caution.')
    mng_group.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'], help='Enable distributed training?')
    mng_group.add_argument('--savecheck', choices=['last', 'task'], type=str, help='Save checkpoint every `task` or at the end of the training (`last`).')
    mng_group.add_argument('--save_checkpoint_mode', choices=['old_pickle', 'safe'], type=str, default='safe',
                           help='Save the model checkpoint with metadata in a single pickle file with the old structure (`old_pickle`) '
                           'or with the new, `safe` structure (default)?. NOTE: the `old_pickle` structure requires `weights_only=False`, which will be '
                           'deprecated by PyTorch.')
    mng_group.add_argument('--loadcheck', type=str, default=None, help='Path of the checkpoint to load (.pt file for the specific task)')
    mng_group.add_argument('--ckpt_name', type=str, help='(optional) checkpoint save name.')
    mng_group.add_argument('--start_from', type=int, default=None, help="Task to start from")
    mng_group.add_argument('--stop_after', type=int, default=None, help="Task limit")

    wandb_group = parser.add_argument_group('Wandb arguments', 'Arguments to manage logging on Wandb.')

    wandb_group.add_argument('--wandb_name', type=str, default=None,
                             help='Wandb name for this run. Overrides the default name (`args.model`).')
    wandb_group.add_argument('--wandb_entity', type=str, help='Wandb entity')
    wandb_group.add_argument('--wandb_project', type=str, help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods

    Args:
        parser: the parser instance

    Returns:
        None
    """
    group = parser.add_argument_group('Rehearsal arguments', 'Arguments shared by all rehearsal-based methods.')

    group.add_argument('--buffer_size', type=int, required=True,
                       help='The size of the memory buffer.')
    group.add_argument('--minibatch_size', type=int,
                       help='The batch size of the memory buffer.')


def check_multiple_defined_arg_during_string_parse() -> None:
    """
    Check if an argument is defined multiple times during the string parsing.
    Prevents the user from typing the same argument multiple times as:
    `--arg1=val1 --arg1=val2`.
    """

    cmd_args = sys.argv[1:]
    keys = set()
    for i, arg in enumerate(cmd_args):
        if '=' in arg:
            arg_name = arg.split('=')[0]
        else:
            arg_name = arg if arg.startswith('-') else None
        if arg_name is not None and arg_name in keys:
            raise ValueError(f"Argument `{arg_name}` is defined multiple times.")
        keys.add(arg_name)


class _DocsArgs:
    """
    This class is used to generate the documentation of the arguments.
    """

    def __init__(self, name: str, tp: str, choices: str, default: str, help_: str):
        if tp is None:
            tp = 'unknown'
        elif tp.__name__ == '_parse_field':
            tp = 'field with aliases (str)'
        elif tp.__name__ == 'binary_to_boolean_type':
            tp = '0|1|True|False -> bool'
        elif tp.__name__ == 'custom_str_underscore':
            tp = 'str (with underscores replaced by dashes)'
        else:
            tp = tp.__name__

        self.name = name
        self.type = tp
        self.choices = choices
        self.default = default
        self.help = help_

    def parse_choices(self) -> str:
        if self.choices is None:
            return ''
        return ', '.join([c.keys() if isinstance(c, dict) else str(c) for c in self.choices])

    def __str__(self):
        tb = f"""**\\-\\-{self.name}** : {self.type}
\t*Help*: {self.help}\n
\t- *Default*: ``{self.default}``"""
        if self.choices is not None:
            tb += f"\n\t- *Choices*: ``{self.parse_choices()}``"
        return tb


class _DocArgsGroup:
    """
    This class is used to generate the documentation of the arguments.
    """

    def __init__(self, group_name: str, group_desc: str, doc_args: _DocsArgs):
        self.group_name = group_name
        self.group_desc = group_desc
        self.doc_args = doc_args

    def __str__(self):
        args_str = '\n'.join([arg.__str__() for arg in self.doc_args])
        s = f""".. rubric:: {self.group_name.capitalize()}\n\n"""
        if self.group_desc:
            s += f"*{self.group_desc}*\n\n"
        s += args_str
        return s


def _parse_actions(actions: list, group_name: str, group_desc: str) -> _DocArgsGroup:
    """
    Parses the actions of the parser.

    Args:
        actions: the actions to parse
        group_name: the name of the group
        group_desc: the description of the group

    Returns:
        an instance of _DocArgsGroup containing the parsed actions
    """
    docs_args = []
    for action in actions:
        if action.dest == 'help':
            continue
        docs_args.append(_DocsArgs(action.dest, action.type, action.choices, action.default, action.help))
    return _DocArgsGroup(group_name, group_desc, docs_args)


if __name__ == '__main__':
    print("Generating documentation for the arguments...")
    os.chdir(mammoth_path)

    parser = ArgumentParser()
    add_initial_args(parser)
    parser.add_argument('--dataset_config', type=str,
                        help='The configuration used for this dataset (e.g., number of tasks, transforms, backbone architecture, etc.).'
                        'The available configurations are defined in the `datasets/config/<dataset>` folder.')
    docs_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        docs_args.append(_DocsArgs(action.dest, action.type, action.choices, action.default, action.help))

    with open('docs/utils/args.rst', 'w') as f:
        f.write('.. _module-args:\n\n')
        f.write('Arguments\n')
        f.write('=========\n\n')
        f.write('.. rubric:: MAIN MAMMOTH ARGS\n\n')
        for arg in docs_args:
            f.write(str(arg) + '\n\n')

    parser = ArgumentParser()
    add_experiment_args(parser)

    docs_args = []
    for group in parser._action_groups:
        if len([a for a in group._group_actions if a.dest != 'help']) == 0:
            continue
        docs_args.append(_parse_actions(group._group_actions, group.title, group.description))

    with open('docs/utils/args.rst', 'a') as f:
        f.write('.. rubric:: EXPERIMENT-RELATED ARGS\n\n')
        for arg in docs_args:
            f.write(str(arg) + '\n\n')

    parser = ArgumentParser()
    add_management_args(parser)
    docs_args = []
    for group in parser._action_groups:
        if len([a for a in group._group_actions if a.dest != 'help']) == 0:
            continue
        docs_args.append(_parse_actions(group._group_actions, group.title, group.description))

    with open('docs/utils/args.rst', 'a') as f:
        f.write('.. rubric:: MANAGEMENT ARGS\n\n')
        for arg in docs_args:
            f.write(str(arg) + '\n\n')

    parser = ArgumentParser()
    add_rehearsal_args(parser)
    docs_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        docs_args.append(_DocsArgs(action.dest, action.type, action.choices, action.default, action.help))

    with open('docs/utils/args.rst', 'a') as f:
        f.write('.. rubric:: REEHARSAL-ONLY ARGS\n\n')
        for arg in docs_args:
            f.write(str(arg) + '\n\n')

    print("Saving documentation in docs/utils/args.rst")
    print("Done!")

    from models import get_model_names

    if os.path.exists('docs/model_args'):
        import shutil
        shutil.rmtree('docs/model_args')
    os.makedirs('docs/model_args')

    for model_name, model_class in get_model_names().items():
        if isinstance(model_class, Exception):
            raise model_class
        try:
            parser = model_class.get_parser(ArgumentParser())
        except Exception as e:
            print('Troubles with model:', model_name)
            raise e

        model_args_groups = []
        for group in parser._action_groups:
            if len([a for a in group._group_actions if a.dest != 'help']) == 0:
                continue
            model_args_groups.append(_parse_actions(group._group_actions, group.title, group.description))
        model_filename = model_name.replace("-", "_")
        with open(f'docs/model_args/{model_filename}_args.rst', 'w') as f:
            f.write(f'Arguments\n')
            f.write(f'~~~~~~~~~~~\n\n')
            for arg in model_args_groups:
                f.write(str(arg) + '\n\n')
        print(f"Saving documentation in docs/model_args/{model_filename}_args.rst")
