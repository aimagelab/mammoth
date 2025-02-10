# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser, Namespace
from typing import Callable, List, Optional, Tuple, Any, TYPE_CHECKING

import torch
import numpy as np
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset

from datasets.utils.label_noise import build_noisy_labels
from datasets.utils.validation import get_validation_indexes
from utils import check_fn_dynamic_type
from utils.conf import create_seeded_dataloader
from datasets.utils import build_torchvision_transform
from utils.loggers import log_accs, log_bias_accs
from utils.prompt_templates import templates
from utils.evaluate import evaluate, EvalFn
from utils.bias import evaluate_with_bias

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel


class MammothDatasetWrapper(Dataset, object):
    """
    Wraps the datasets used inside the ContinualDataset class to allow for a more flexible retrieval of the data.
    """
    data: np.ndarray  # Required: the data of the dataset
    targets: np.ndarray  # Required: the targets of the dataset
    indexes: np.ndarray  # The original indexes of the items in the complete dataset
    task_ids: np.ndarray  # The corresponding task ids of the items in the complete dataset. If present, will be used to split the dataset into tasks

    required_fields = ('data', 'targets')  # Required: the fields that must be defined
    extra_return_fields: Tuple[str] = tuple()  # Optional: extra fields to return from the dataset (must be defined)

    is_init: bool = False

    def __getattr__(self, name: str) -> Any:
        if self.is_init and hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        if name not in vars(self):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.is_init and hasattr(self.dataset, name):
            return setattr(self.dataset, name, value)
        return super().__setattr__(name, value)

    def __hasattr__(self, name: str) -> bool:
        if self.is_init and name == '__getitem__' or name == '__len__':
            return hasattr(self.dataset, name)
        return super().__hasattr__(name)

    def __init__(self, ext_dataset: Dataset, train: bool = False):
        super().__init__()

        self.dataset = ext_dataset
        self.train = train
        missing_fields = [field for field in self.required_fields if not hasattr(self.dataset, field)]
        assert len(missing_fields) == 0, 'The dataset is missing some required fields:' + ', '.join(missing_fields)

        self.indexes = np.arange(len(self.dataset))

        self._c_iter = 0
        self.num_times_iterated = 0
        self.is_init = True

    def __len__(self):
        return len(self.dataset)

    def add_extra_return_field(self, field_name: str, field_value) -> None:
        """
        Adds an extra field to the dataset.

        Args:
            field_name (str): the name of the field
            field_value: the value of the field
        """
        setattr(self, field_name, field_value)
        self.extra_return_fields += (field_name,)

    def extend_return_items(self, ret_tuple: Tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]],
                            index: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Tuple[Optional[torch.Tensor]]]:
        """
        Extends the return tuple with the extra fields defined in `extra_return_fields`.

        Args:
            ret_tuple (Tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]): the current return tuple

        Returns:
            Tuple[torch.Tensor, int, Optional[torch.Tensor], Sequence[Optional[torch.Tensor]]]: the extended return tuple
        """
        tmp_tuple = []
        for name in self.extra_return_fields:
            attr = getattr(self, name)
            c_idx = index if len(attr) == len(self.data) else self.indexes[index]
            attr = attr[c_idx]
            tmp_tuple.append(attr)

        ret_tuple = list(ret_tuple) + tmp_tuple

        return tuple(ret_tuple)

    def __iter__(self):
        self._c_iter = 0
        self.num_times_iterated += 1
        return iter(self.dataset)

    def __next__(self):
        ret_tuple = next(self.dataset)
        ret_tuple = self.extend_return_items(ret_tuple, self._c_iter)
        self._c_iter += 1
        return ret_tuple

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]:
        ret_tuple = self.dataset.__getitem__(index)
        ret_tuple = self.extend_return_items(ret_tuple, index)

        return ret_tuple


class ContinualDataset(object):
    """
    A base class for defining continual learning datasets.

    Data is divided into tasks and loaded only when the `get_data_loaders` method is called.

    Attributes:
        NAME (str): the name of the dataset
        SETTING (str): the setting of the dataset
        N_CLASSES_PER_TASK (int): the number of classes per task
        N_TASKS (int): the number of tasks
        N_CLASSES (int): the number of classes
        SIZE (Tuple[int]): the size of the dataset
        AVAIL_SCHEDS (List[str]): the available schedulers
        class_names (List[str]): list of the class names of the dataset (should be populated by `get_class_names`)
        train_loader (DataLoader): the training loader
        test_loaders (List[DataLoader]): the test loaders
        i (int): the current task
        c_task (int): the current task
        args (Namespace): the arguments which contains the hyperparameters
        eval_fn (Callable): the function used to evaluate the model on the dataset
    """

    base_fields = ('SETTING', 'N_CLASSES_PER_TASK', 'N_TASKS', 'SIZE', 'N_CLASSES', 'AVAIL_SCHEDS')
    optional_fields = ('MEAN', 'STD')
    composed_fields = {
        'TRANSFORM': build_torchvision_transform,
        'TEST_TRANSFORM': build_torchvision_transform
    }

    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    N_CLASSES: int
    SIZE: Tuple[int]
    AVAIL_SCHEDS = ['multisteplr', 'cosine']
    class_names: List[str] = None
    eval_fn: EvalFn
    log_fn: Callable

    @property
    def current_task(self) -> int:
        """
        Returns the current task index.
        """
        if self.c_task == -1 and self.SETTING in ['task-il', 'class-il']:
            raise ValueError('The dataset has not been initialized yet.')
        return len(self.test_loaders)  # self.c_task is not updated in the case of domain-il

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.

        Args:
            args: the arguments which contains the hyperparameters
        """
        if self.SETTING == 'biased-class-il':
            self.eval_fn = evaluate_with_bias
            self.log_fn = log_bias_accs
        else:
            self.eval_fn = evaluate
            self.log_fn = log_accs

        self.train_loader = None
        self.test_loaders = []
        self.c_task = -1
        self.args = args
        if 'class-il' in self.SETTING:
            self.N_CLASSES = self.N_CLASSES if hasattr(self, 'N_CLASSES') else \
                (self.N_CLASSES_PER_TASK * self.N_TASKS) if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK)
        else:
            self.N_CLASSES = self.N_CLASSES_PER_TASK

        if self.args.permute_classes:
            if not hasattr(self.args, 'class_order'):  # set only once
                if self.args.seed is not None:
                    np.random.seed(self.args.seed)
                self.args.class_order = np.random.permutation(self.N_CLASSES)

        if args.label_perc != 1 or args.label_perc_by_class != 1:
            self.unlabeled_rng = np.random.RandomState(args.seed)

        if args.joint:
            if self.SETTING in ['class-il', 'task-il']:
                # just set the number of classes per task to the total number of classes
                self.N_CLASSES_PER_TASK = self.N_CLASSES
                self.N_TASKS = 1
            else:
                # bit more tricky, not supported for now
                raise NotImplementedError('Joint training is only supported for class-il and task-il.'
                                          'For other settings, please use the `joint` model with `--model=joint` and `--joint=0`')

        missing_fields = [field for field in self.base_fields if not hasattr(self, field) or getattr(self, field) is None]
        if len(missing_fields) > 0:
            raise NotImplementedError('The dataset must be initialized with all the required fields but is missing:', missing_fields)

    def log(self, *args, **kwargs):
        """
        Logs the result of the evaluation.

        By default, it logs the accuracy of the model on the current task using the `utils.loggers.log_accs` function.
        """

        return self.log_fn(*args, **kwargs)

    @staticmethod
    def evaluate(model: 'ContinualModel', dataset: 'ContinualDataset', last=False, return_loss=False):
        """
        Evaluates the model on the current task.

        Args:
            model: the model to be evaluated
            dataset: the continual dataset at hand (optional). If not provided, the current dataset (self) will be used
            last: a boolean indicating whether to evaluate only the last task
            return_loss: a boolean indicating whether to return the loss in addition to the accuracy

        Returns:
            tuple: the evaluation results
        """

        check_fn_dynamic_type(dataset.eval_fn, EvalFn)
        return dataset.eval_fn(model, dataset, last=last, return_loss=return_loss)

    @classmethod
    def set_default_from_config(cls, config: dict, parser: ArgumentParser) -> dict:
        """
        Sets the default arguments from the configuration file.
        The default values will be set in the class attributes and will be available for all instances of the class.

        The arguments that are related to the dataset (i.e., are in the 'base_fields', 'optional_fields', or 'composed_fields') will be removed from the config dictionary to avoid conflicts with the command line arguments.

        Args:
            config (dict): the configuration file
            parser (ArgumentParser): the argument parser to set the default values

        Returns:
            dict: the configuration file without the dataset-related arguments
        """

        tmp_config = config.copy()

        _base_fields = [k.casefold() for k in cls.base_fields]
        _optional_fields = [k.casefold() for k in cls.optional_fields]
        _composed_fields = [k.casefold() for k in cls.composed_fields.keys()]

        for k, v in config.items():
            if k.casefold() in _base_fields:
                _k = cls.base_fields[_base_fields.index(k.casefold())]
                setattr(cls, _k, v)
                del tmp_config[k]
            elif k.casefold() in _optional_fields:
                k = cls.optional_fields[_optional_fields.index(k.casefold())]
                setattr(cls, k, v)
                del tmp_config[k]
            elif k.casefold() in _composed_fields:
                _k = list(cls.composed_fields.keys())[_composed_fields.index(k.casefold())]
                setattr(cls, _k, cls.composed_fields[_k](v))
                del tmp_config[k]
            else:
                setattr(cls, k, v)
                parser.set_defaults(**{k: v})

        return tmp_config

    def get_offsets(self, task_idx: int = None):
        """
        Compute the start and end class index for the current task.

        Args:
            task_idx (int): the task index

        Returns:
            tuple: the start and end class index for the current task
        """
        if self.SETTING == 'class-il' or self.SETTING == 'task-il':
            task_idx = task_idx if task_idx is not None else self.c_task
        else:
            task_idx = 0

        start_c = self.N_CLASSES_PER_TASK * task_idx if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK[:task_idx])
        end_c = self.N_CLASSES_PER_TASK * (task_idx + 1) if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK[:task_idx + 1])

        assert end_c > start_c, 'End class index must be greater than start class index.'

        return start_c, end_c

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.

        Returns:
            the current training and test loaders
        """
        raise NotImplementedError

    def get_backbone(self) -> str:
        """Returns the name of the backbone to be used for the current dataset. This can be changes using the `--backbone` argument or by setting it in the `dataset_config`."""
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """Returns the transform to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """Returns the loss to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """Returns the transform used for normalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """Returns the transform used for denormalizing the current dataset."""
        raise NotImplementedError

    def get_iters(self):
        """Returns the number of iterations to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_iters` to set the default number of iterations.')

    def get_epochs(self):
        """Returns the number of epochs to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_epochs` to set the default number of epochs.')

    def get_batch_size(self):
        """Returns the batch size to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_batch_size` to set the default batch size.')

    def get_minibatch_size(self):
        """Returns the minibatch size to be used for the current dataset."""
        return self.get_batch_size()

    def get_class_names(self) -> List[str]:
        """Returns the class names for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_class_names` to get the class names.')

    def get_prompt_templates(self) -> List[str]:
        """
        Returns the prompt templates for the current dataset.
        By default, it returns the ImageNet prompt templates.
        """
        return templates['imagenet']


def _get_mask_unlabeled(train_dataset, setting: ContinualDataset):
    if setting.args.label_perc == 1 and setting.args.label_perc_by_class == 1:
        return np.zeros(train_dataset.targets.shape[0]).astype('bool')
    else:
        if setting.args.label_perc != 1:  # label perc by task
            lpt = int(setting.args.label_perc * (train_dataset.targets.shape[0] // setting.N_CLASSES_PER_TASK))
            ind = np.indices(train_dataset.targets.shape)[0]
            mask = []
            for lab in np.unique(train_dataset.targets):
                partial_targets = train_dataset.targets[train_dataset.targets == lab]
                current_mask = setting.unlabeled_rng.choice(partial_targets.shape[0], max(
                    partial_targets.shape[0] - lpt, 0), replace=False)

                mask.append(ind[train_dataset.targets == lab][current_mask])
        else:  # label perc by class
            unique_labels, label_count_by_class = np.unique(train_dataset.targets, return_counts=True)
            lpcs = (setting.args.label_perc_by_class * label_count_by_class).astype(np.int32)
            mask = []
            for lab, count, lpc in zip(unique_labels, label_count_by_class, lpcs):
                current_mask = setting.unlabeled_rng.choice(count, max(count - lpc, 0), replace=False)
                mask.append(np.where(train_dataset.targets == lab)[0][current_mask])

        return np.array(mask).astype(np.int32)


def _prepare_data_loaders(train_dataset: MammothDatasetWrapper, test_dataset: MammothDatasetWrapper, setting: ContinualDataset):
    if isinstance(train_dataset.targets, list) or not train_dataset.targets.dtype is torch.long:
        train_dataset.targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    if isinstance(test_dataset.targets, list) or not test_dataset.targets.dtype is torch.long:
        test_dataset.targets = torch.tensor(test_dataset.targets, dtype=torch.long)

    setting.unlabeled_mask = _get_mask_unlabeled(train_dataset, setting)

    if setting.unlabeled_mask.sum() != 0:
        train_dataset.targets[setting.unlabeled_mask] = -1  # -1 is the unlabeled class

    return train_dataset, test_dataset


def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.

    Attributes:
        train_dataset (Dataset): the training dataset
        test_dataset (Dataset): the test dataset
        setting (ContinualDataset): the setting of the dataset

    Returns:
        the training and test loaders
    """
    # Initializations
    train_dataset = MammothDatasetWrapper(train_dataset, train=True)
    test_dataset = MammothDatasetWrapper(test_dataset, train=False)

    if 'class-il' in setting.SETTING or 'task-il' in setting.SETTING:
        setting.c_task += 1

    if not isinstance(train_dataset.targets, np.ndarray):
        train_dataset.targets = np.array(train_dataset.targets)
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    # Permute classes
    if setting.args.permute_classes:
        train_dataset.targets = setting.args.class_order[train_dataset.targets]
        test_dataset.targets = setting.args.class_order[test_dataset.targets]

    # Setup validation
    if setting.args.validation:
        train_idxs, val_idxs = get_validation_indexes(setting.args.validation, train_dataset, setting.args.seed)

        test_dataset.data = train_dataset.data[val_idxs]
        test_dataset.targets = train_dataset.targets[val_idxs]
        test_dataset.indexes = train_dataset.indexes[val_idxs]

        train_dataset.data = train_dataset.data[train_idxs]
        train_dataset.targets = train_dataset.targets[train_idxs]
        train_dataset.indexes = train_dataset.indexes[train_idxs]

    # Apply noise to the labels
    if setting.args.noise_rate > 0:
        train_dataset.add_extra_return_field('true_labels', train_dataset.targets.copy())  # save original targets before adding noise
        noisy_targets = build_noisy_labels(train_dataset.targets, setting.args)
        train_dataset.targets = noisy_targets  # overwrite the targets with the noisy ones

    # Split the dataset into tasks
    if 'class-il' in setting.SETTING or 'task-il' in setting.SETTING:
        if hasattr(train_dataset, 'task_ids'):
            if not isinstance(test_dataset.task_ids, np.ndarray):
                test_dataset.task_ids = np.array(test_dataset.task_ids)
            if not isinstance(train_dataset.task_ids, np.ndarray):
                train_dataset.task_ids = np.array(train_dataset.task_ids)

            train_mask = train_dataset.task_ids == setting.c_task

            if setting.args.validation_mode == 'current':
                test_mask = test_dataset.task_ids == setting.c_task
            elif setting.args.validation_mode == 'complete':
                test_mask = np.logical_and(test_dataset.task_ids >= 0, test_dataset.task_ids <= setting.c_task)
            else:
                raise ValueError('Unknown validation mode: {}'.format(setting.args.validation_mode))
        else:
            start_c, end_c = setting.get_offsets()

            train_mask = np.logical_and(train_dataset.targets >= start_c,
                                        train_dataset.targets < end_c)

            if setting.args.validation_mode == 'current':
                test_mask = np.logical_and(test_dataset.targets >= start_c,
                                           test_dataset.targets < end_c)
            elif setting.args.validation_mode == 'complete':
                test_mask = np.logical_and(test_dataset.targets >= 0,
                                           test_dataset.targets < end_c)
            else:
                raise ValueError('Unknown validation mode: {}'.format(setting.args.validation_mode))

        test_dataset.data = test_dataset.data[test_mask]
        test_dataset.targets = test_dataset.targets[test_mask]
        test_dataset.indexes = test_dataset.indexes[test_mask]
        if hasattr(test_dataset, 'task_ids'):
            test_dataset.task_ids = test_dataset.task_ids[test_mask]

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = train_dataset.targets[train_mask]
        train_dataset.indexes = train_dataset.indexes[train_mask]
        if hasattr(train_dataset, 'task_ids'):
            train_dataset.task_ids = train_dataset.task_ids[train_mask]

        if setting.SETTING == 'biased-class-il':
            assert hasattr(test_dataset, 'bias_label'), 'The dataset must have the bias label field (used during evaluation).'
            train_dataset.add_extra_return_field('indexes', np.arange(len(train_dataset.targets)))
            test_dataset.add_extra_return_field('bias_label', test_dataset.bias_label[test_mask])

    # Finalize data, apply unlabeled mask
    train_dataset, test_dataset = _prepare_data_loaders(train_dataset, test_dataset, setting)

    # Create dataloaders
    train_loader = create_seeded_dataloader(setting.args, train_dataset,
                                            batch_size=setting.args.batch_size, shuffle=True, drop_last=setting.args.drop_last)
    test_loader = create_seeded_dataloader(setting.args, test_dataset,
                                           batch_size=setting.args.batch_size, shuffle=False)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def fix_class_names_order(class_names: List[str], args: Namespace) -> List[str]:
    """
    Permutes the order of the class names according to the class order specified in the arguments.
    The order reflects that of `store_masked_loaders`.

    Args:
        class_names: the list of class names. This should contain all classes in the dataset (not just the current task's ones).
        args: the command line arguments

    Returns:
        List[str]: the class names in the correct order
    """
    if args.permute_classes:
        class_names = [class_names[np.where(args.class_order == i)[0][0]] for i in range(len(class_names))]
    return class_names
