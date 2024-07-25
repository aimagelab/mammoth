# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
if __name__ == '__main__':
    import os
    import sys
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(mammoth_path)

from argparse import ArgumentParser
from datasets import get_dataset_names
from models import get_all_models
from models.utils.continual_model import ContinualModel
from utils import custom_str_underscore


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    exp_group = parser.add_argument_group('Experiment arguments', 'Arguments used to define the experiment settings.')

    exp_group.add_argument('--dataset', type=str, required=True,
                           choices=get_dataset_names(),
                           help='Which dataset to perform experiments on.')
    exp_group.add_argument('--model', type=custom_str_underscore, required=True,
                           help='Model name.', choices=list(get_all_models().keys()))
    exp_group.add_argument('--lr', type=float, required=True, help='Learning rate.')
    exp_group.add_argument('--batch_size', type=int, help='Batch size.')
    exp_group.add_argument('--label_perc', type=float, default=1, help='Percentage in (0-1] of labeled examples per task.')
    exp_group.add_argument('--joint', type=int, choices=(0, 1), default=0, help='Train model on Joint (single task)?')
    exp_group.add_argument('--eval_future', type=int, choices=(0, 1), default=0, help='Evaluate future tasks?')

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
    opt_group.add_argument('--optim_nesterov', type=int, default=0,
                           help='optimizer nesterov momentum.')
    opt_group.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler.')
    opt_group.add_argument('--lr_milestones', type=int, nargs='+', default=[],
                           help='Learning rate scheduler milestones (used if `lr_scheduler=multisteplr`).')
    opt_group.add_argument('--sched_multistep_lr_gamma', type=float, default=0.1,
                           help='Learning rate scheduler gamma (used if `lr_scheduler=multisteplr`).')


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
    mng_group.add_argument('--permute_classes', type=int, choices=[0, 1], default=1,
                           help='Permute classes before splitting into tasks? This applies the seed before permuting if the `seed` argument is present.')
    mng_group.add_argument('--base_path', type=str, default="./data/",
                           help='The base path where to save datasets, logs, results.')
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
    mng_group.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    mng_group.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Disable logging?')
    mng_group.add_argument('--num_workers', type=int, default=None, help='Number of workers for the dataloaders (default=infer from number of cpus).')
    mng_group.add_argument('--enable_other_metrics', default=0, choices=[0, 1], type=int,
                           help='Enable computing additional metrics: forward and backward transfer.')
    mng_group.add_argument('--debug_mode', type=int, default=0, choices=[0, 1], help='Run only a few training steps per epoch. This also disables logging on wandb.')
    mng_group.add_argument('--inference_only', default=0, choices=[0, 1], type=int,
                           help='Perform inference only for each task (no training).')
    mng_group.add_argument('-O', '--code_optimization', type=int, default=0, choices=[0, 1, 2, 3],
                           help='Optimization level for the code.'
                           '0: no optimization.'
                           '1: Use TF32, if available.'
                           '2: Use BF16, if available.'
                           '3: Use BF16 and `torch.compile`. BEWARE: torch.compile may break your code if you change the model after the first run! Use with caution.')
    mng_group.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'], help='Enable distributed training?')
    mng_group.add_argument('--savecheck', choices=['last', 'task'], type=str, help='Save checkpoint every `task` or at the end of the training (`last`).')
    mng_group.add_argument('--loadcheck', type=str, default=None, help='Path of the checkpoint to load (.pt file for the specific task)')
    mng_group.add_argument('--ckpt_name', type=str, required=False, help='(optional) checkpoint save name.')
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


class _DocsArgs:
    """
    This class is used to generate the documentation of the arguments.
    """

    def __init__(self, name: str, type_: str, choices: str, default: str, help_: str):
        self.name = name
        self.type = type_
        self.choices = choices
        self.default = default
        self.help = help_

    def parse_choices(self) -> str:
        if self.choices is None:
            return ''
        return ', '.join([c.keys() if isinstance(c, dict) else str(c) for c in self.choices])

    def __str__(self):
        tb = f"""**\\-\\-{self.name}** : {self.type.__name__ if self.type is not None else 'unknown'}
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
    add_experiment_args(parser)

    docs_args = []
    for group in parser._action_groups:
        if len([a for a in group._group_actions if a.dest != 'help']) == 0:
            continue
        docs_args.append(_parse_actions(group._group_actions, group.title, group.description))

    with open('docs/utils/args.rst', 'w') as f:
        f.write('.. _module-args:\n\n')
        f.write('Arguments\n')
        f.write('=========\n\n')
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

    for model_name, model_class in get_model_names().items():
        parser = model_class.get_parser()

        model_args_groups = []
        for group in parser._action_groups:
            if len([a for a in group._group_actions if a.dest != 'help']) == 0:
                continue
            model_args_groups.append(_parse_actions(group._group_actions, group.title, group.description))
        model_filename = model_name.replace("-", "_")
        with open(f'docs/models/{model_filename}_args.rst', 'w') as f:
            f.write(f'Arguments\n')
            f.write(f'~~~~~~~~~~~\n\n')
            for arg in model_args_groups:
                f.write(str(arg) + '\n\n')
        print(f"Saving documentation in docs/models/{model_filename}_args.rst")
