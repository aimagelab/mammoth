"""
This is the base class for all models. It provides some useful methods and defines the interface of the models.

The `observe` method is the most important one: it is called at each training iteration and it is responsible for computing the loss and updating the model's parameters.

The `begin_task` and `end_task` methods are called before and after each task, respectively.

The `get_parser` method returns the parser of the model. Additional model-specific hyper-parameters can be added by overriding this method.

The `get_debug_iters` method returns the number of iterations to be used for debugging. Default: 3.

The `get_optimizer` method returns the optimizer to be used for training. Default: SGD.

The `load_buffer` method is called when a buffer is loaded. Default: do nothing.

The `meta_observe`, `meta_begin_task` and `meta_end_task` methods are wrappers for `observe`, `begin_task` and `end_task` methods, respectively. They take care of updating the internal counters and of logging to wandb if installed.

The `autolog_wandb` method is used to automatically log to wandb all variables starting with "_wandb_" or "loss" in the observe function. It is called by `meta_observe` if wandb is installed. It can be overridden to add custom logging.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
import logging
import sys
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
import inspect

import kornia
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset

from utils.buffer import Buffer
from utils.conf import get_device, warn_once
from utils.kornia_utils import to_kornia_transform
from utils.magic import persistent_locals
from torchvision import transforms

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]
    AVAIL_OPTIMS = ['sgd', 'adam', 'adamw']

    args: Namespace  # The command line arguments
    device: torch.device  # The device to be used for training
    net: 'MammothBackbone'  # The backbone of the model (defined by the `dataset`)
    loss: nn.Module  # The loss function to be used (defined by the `dataset`)
    opt: optim.Optimizer  # The optimizer to be used for training
    scheduler: optim.lr_scheduler._LRScheduler  # (optional) The scheduler for the optimizer. If defined, it will overwrite the one defined in the `dataset`
    # The transformation to be applied to the input data. The model will try to convert it to a kornia transform to be applicable to a batch of samples at once
    transform: Union[transforms.Compose, kornia.augmentation.AugmentationSequential]
    original_transform: transforms.Compose  # The original transformation to be applied to the input data. This is the one defined by the `dataset`
    task_iteration: int  # Number of iterations in the current task
    epoch_iteration: int  # Number of iterations in the current epoch. Updated if `epoch` is passed to observe
    dataset: 'ContinualDataset'  # The instance of the dataset. Used to update the number of classes in the current task
    num_classes: int  # Total number of classes in the dataset
    n_tasks: int  # Total number of tasks in the dataset

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines model-specific hyper-parameters, which will be added to the command line arguments. Additional model-specific hyper-parameters can be added by overriding this method.

        For backward compatibility, the `parser` object may be omitted (although this should be avoided). In this case, the method should create and return a new parser.

        This method may also be used to set default values for all other hyper-parameters of the framework (e.g., `lr`, `buffer_size`, etc.) with the `set_defaults` method of the parser. In this case, this method MUST update the original `parser` object and not create a new one.

        Args:
            parser: the main parser, to which the model-specific arguments will be added

        Returns:
            the parser of the model
        """
        return parser

    @property
    def task_iteration(self):
        """
        Returns the number of iterations in the current task.
        """
        return self._task_iteration

    @property
    def epoch_iteration(self):
        """
        Returns the number of iterations in the current epoch.
        """
        return self._epoch_iteration

    @property
    def current_task(self):
        """
        Returns the index of current task.
        """
        return self._current_task

    @property
    def n_classes_current_task(self):
        """
        Returns the number of classes in the current task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_classes_current_task'):
            return self._n_classes_current_task
        else:
            return -1

    @property
    def n_seen_classes(self):
        """
        Returns the number of classes seen so far.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_seen_classes'):
            return self._n_seen_classes
        else:
            return -1

    @property
    def n_remaining_classes(self):
        """
        Returns the number of classes remaining to be seen.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_remaining_classes'):
            return self._n_remaining_classes
        else:
            return -1

    @property
    def n_past_classes(self):
        """
        Returns the number of classes seen up to the PAST task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_past_classes'):
            return self._n_past_classes
        else:
            return -1

    @property
    def cpt(self):
        """
        Alias of `classes_per_task`: returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers depending on the dataset.
        """
        return self._cpt

    @property
    def classes_per_task(self):
        """
        Returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers depending on the dataset.
        """
        return self._cpt

    @cpt.setter
    def cpt(self, value):
        """
        Sets the number of classes per task.
        """
        warn_once("Setting the number of classes per task is not recommended.")
        self._cpt = value

    def __init__(self, backbone: 'MammothBackbone', loss: nn.Module,
                 args: Namespace, transform: nn.Module, dataset: 'ContinualDataset' = None) -> None:
        super(ContinualModel, self).__init__()
        print("Using {} as backbone".format(backbone.__class__.__name__))
        self.net = backbone
        self.loss = loss
        self.args = args
        self.original_transform = transform
        self.transform = transform
        if dataset is None:
            logging.error("No dataset provided. Will create another instance but NOTE that this **WILL** result in some bugs and possible memory leaks!")
            self.dataset = get_dataset(self.args)
        else:
            self.dataset = dataset
        self.N_CLASSES = self.dataset.N_CLASSES
        self.num_classes = self.N_CLASSES
        self.N_TASKS = self.dataset.N_TASKS
        self.n_tasks = self.N_TASKS
        self.SETTING = self.dataset.SETTING
        self._cpt = self.dataset.N_CLASSES_PER_TASK
        self._current_task = 0

        try:
            self.transform = to_kornia_transform(transform.transforms[-1].transforms)
            self.normalization_transform = to_kornia_transform(self.dataset.get_normalization_transform())
        except BaseException:
            logging.error("could not initialize kornia transforms.")
            self.normalization_transform = transforms.Compose([transforms.ToPILImage(), self.dataset.TEST_TRANSFORM]) if hasattr(
                self.dataset, 'TEST_TRANSFORM') else transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), self.dataset.get_normalization_transform()])

        if self.net is not None:
            self.opt = self.get_optimizer()
        else:
            logging.warning("no default model for this dataset. You will have to specify the optimizer yourself.")
            self.opt = None
        self.device = get_device()

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

        if self.args.label_perc != 1 and 'cssl' not in self.COMPATIBILITY:
            logging.info('label_perc is not explicitly supported by this model -> training may break')

    def to(self, device):
        """
        Captures the device to be used for training.
        """
        self.device = device
        return super().to(device)

    def load_buffer(self, buffer):
        """
        Default way to handle load buffer.
        """

        if isinstance(buffer, Buffer):
            assert buffer.examples.shape[0] == self.args.buffer_size, "Buffer size mismatch. Expected {} got {}".format(
                self.args.buffer_size, buffer.examples.shape[0])
            self.buffer = buffer
        elif isinstance(buffer, dict):  # serialized buffer
            assert 'examples' in buffer, "Buffer does not contain examples"
            assert self.buffer.buffer_size == buffer['examples'].shape[0], "Buffer size mismatch. Expected {} got {}".format(
                self.buffer.buffer_size, buffer['examples'].shape[0])
            for k, v in buffer.items():
                setattr(self.buffer, k, v)
            self.buffer.attributes = list(buffer.keys())
            self.buffer.num_seen_examples = buffer['examples'].shape[0]
        else:
            raise ValueError("Buffer type not recognized")

    def get_parameters(self):
        """
        Returns the parameters of the model.
        """
        return self.net.parameters()

    def get_optimizer(self, params: Iterator[torch.Tensor] = None, lr=None) -> optim.Optimizer:
        """
        Returns the optimizer to be used for training.

        Default: SGD.

        Args:
            params: the parameters to be optimized. If None, the default specified by `get_parameters` is used.
            lr: the learning rate. If None, the default specified by the command line arguments is used.

        Returns:
            the optimizer
        """

        params = params if params is not None else self.get_parameters()
        if params is None:
            logging.info("No parameters to optimize.")
            return None
        params = list(params)
        if len(params) == 0:
            logging.info("No parameters to optimize.")
            return None

        lr = lr if lr is not None else self.args.lr
        # check if optimizer is in torch.optim
        supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(params, lr=lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(params, lr=lr,
                                                                                    weight_decay=self.args.optim_wd)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def compute_offsets(self, task: int) -> Tuple[int, int]:
        """
        Compute the start and end offset given the task.

        Args:
            task: the task index

        Returns:
            the start and end offset
        """
        return self.dataset.get_offsets(task)

    def get_debug_iters(self):
        """
        Returns the number of iterations to be used for debugging.
        Default: 3
        """
        return 5

    def begin_task(self, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the current task.
        Executed before each task.
        """
        pass

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the next task.
        Executed after each task.
        """
        pass

    def begin_epoch(self, epoch: int, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the current epoch.
        """
        pass

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the next task.
        Executed after each task.
        """
        pass

    def end_epoch(self, epoch: int, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the next epoch.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.

        Args:
            x: batch of inputs
            task_label: some models require the task label

        Returns:
            the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        """
        Wrapper for `observe` method.

        Takes care of dropping unlabeled data if not supported by the model and of logging to wandb if installed.

        Args:
            inputs: batch of inputs
            labels: batch of labels
            not_aug_inputs: batch of inputs without augmentation
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        if 'epoch' in kwargs and kwargs['epoch'] is not None:
            epoch = kwargs['epoch']
            if self._past_epoch != epoch:
                self._past_epoch = epoch
                self._epoch_iteration = 0

        if 'cssl' not in self.COMPATIBILITY:  # drop unlabeled data if not supported
            labeled_mask = args[1] != -1
            if (~labeled_mask).any():  # if there are any unlabeled samples
                if labeled_mask.sum() == 0:  # if all samples are unlabeled
                    return 0
                args = [arg[labeled_mask] if isinstance(arg, torch.Tensor) and arg.shape[0] == args[0].shape[0] else arg for arg in args]

        # remove kwargs that are not needed by the observe method
        observe_args = inspect.signature(self.observe).parameters.keys()
        if 'kwargs' not in observe_args:
            kwargs = {k: v for k, v in kwargs.items() if k in observe_args}

        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            extra = {}
            if isinstance(ret, dict):
                assert 'loss' in ret, "Loss not found in return dict"
                extra = {k: v for k, v in ret.items() if k != 'loss'}
                ret = ret['loss']
            self.autolog_wandb(pl.locals, extra=extra)
        else:
            ret = self.observe(*args, **kwargs)
            if isinstance(ret, dict):
                assert 'loss' in ret, "Loss not found in return dict"
                ret = ret['loss']
        self._task_iteration += 1
        self._epoch_iteration += 1
        return ret

    def meta_begin_task(self, dataset):
        """
        Wrapper for `begin_task` method.

        Takes care of updating the internal counters.

        Args:
            dataset: the current task's dataset
        """
        # update internal counters
        self._task_iteration = 0
        self._epoch_iteration = 0
        self._past_epoch = 0
        self._n_classes_current_task = self._cpt if isinstance(self._cpt, int) else self._cpt[self._current_task]
        self._n_past_classes, self._n_seen_classes = self.compute_offsets(self._current_task)
        self._n_remaining_classes = self.N_CLASSES - self._n_seen_classes

        # reload optimizer if the model has no scheduler
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            if hasattr(self, 'opt') and self.opt is not None:
                self.opt.zero_grad(set_to_none=True)
            self.opt = self.get_optimizer()
        else:
            logging.warning("Model defines a custom scheduler. The optimizer will not be reloaded.")

        # call the actual method
        self.begin_task(dataset)

    def meta_end_task(self, dataset):
        """
        Wrapper for `end_task` method.

        Takes care of updating the internal counters.

        Args:
            dataset: the current task's dataset
        """

        self.end_task(dataset)
        self._current_task += 1

    @abstractmethod
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step over a given batch of examples.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals, extra=None):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            tmp = {k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                   for k, v in locals.items() if k.startswith('_wandb_') or 'loss' in k.lower()}
            tmp.update(extra or {})
            if hasattr(self, 'opt'):
                tmp['lr'] = self.opt.param_groups[0]['lr']
            wandb.log(tmp)
