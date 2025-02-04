# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from utils.augmentations import apply_transform
from utils.conf import create_seeded_dataloader, get_device

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone


def icarl_replay(self: 'ContinualModel', dataset: 'ContinualDataset', val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    Args:
        self: the model instance
        dataset: the dataset
        val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.current_task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_dataset = deepcopy(dataset.train_loader.dataset)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_dataset.targets = np.concatenate([
                self.val_dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_dataset.data = data_concatenate([
                self.val_dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])

            self.val_loader = create_seeded_dataloader(self.args, self.val_dataset, batch_size=self.args.batch_size, shuffle=True)


class BaseSampleSelection:
    """
    Base class for sample selection strategies.
    """

    def __init__(self, buffer_size: int, device):
        """
        Initialize the sample selection strategy.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the buffer on
        """
        self.buffer_size = buffer_size
        self.device = device

    def __call__(self, num_seen_examples: int) -> int:
        """
        Selects the index of the sample to replace.

        Args:
            num_seen_examples: the number of seen examples

        Returns:
            the index of the sample to replace
        """

        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        (optional) Update the state of the sample selection strategy.
        """
        pass


class ReservoirSampling(BaseSampleSelection):
    def __call__(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1


class BalancoirSampling(BaseSampleSelection):
    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        self.unique_map = np.empty((0,), dtype=np.int32)

    def update_unique_map(self, label_in, label_out=None):
        while len(self.unique_map) <= label_in:
            self.unique_map = np.concatenate((self.unique_map, np.zeros((len(self.unique_map) * 2 + 1), dtype=np.int32)), axis=0)
        self.unique_map[label_in] += 1
        if label_out is not None:
            self.unique_map[label_out] -= 1

    def __call__(self, num_seen_examples: int, labels: torch.Tensor, proposed_class: int) -> int:
        """
        Balancoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size
            labels: the set of buffer labels
            proposed_class: the class of the current example

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < self.buffer_size or len(self.unique_map) <= proposed_class or self.unique_map[proposed_class] < np.median(
                self.unique_map[self.unique_map > 0]):
            target_class = np.argmax(self.unique_map)
            # e = rand % self.unique_map.max()
            idx = np.arange(self.buffer_size)[labels.cpu() == target_class][rand % self.unique_map.max()]
            return idx
        else:
            return -1


class LARSSampling(BaseSampleSelection):
    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values

    def normalize_scores(self, values: torch.Tensor):
        if values.shape[0] > 0:
            if values.max() - values.min() != 0:
                values = (values - values.min()) / ((values.max() - values.min()) + 1e-9)
            return values
        else:
            return None

    def __call__(self, num_seen_examples: int) -> int:
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            norm_importance = self.normalize_scores(self.importance_scores)
            norm_importance = norm_importance / (norm_importance.sum() + 1e-9)
            index = np.random.choice(range(self.buffer_size), p=norm_importance.cpu().numpy(), size=1)
            return index
        else:
            return -1


class LossAwareBalancedSampling(BaseSampleSelection):
    """
    Combination of Loss-Aware Sampling (LARS) and Balanced Reservoir Sampling (BRS) from `Rethinking Experience Replay: a Bag of Tricks for Continual Learning`.
    """

    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')
        # balancoir scores
        self.balance_scores = torch.ones(self.buffer_size, dtype=torch.float).to(self.device) * -float('inf')
        # merged scores
        self.scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values

    def merge_scores(self):
        scaling_factor = self.importance_scores.abs().mean() * self.balance_scores.abs().mean()
        norm_importance = self.importance_scores / scaling_factor
        presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores

        if presoftscores.max() - presoftscores.min() != 0:
            presoftscores = (presoftscores - presoftscores.min()) / (presoftscores.max() - presoftscores.min() + 1e-9)
        self.scores = presoftscores / presoftscores.sum()

    def update_balancoir_scores(self, labels: torch.Tensor):
        unique_labels, orig_inputs_idxs, counts = labels.unique(return_counts=True, return_inverse=True)
        # assert len(counts) > unique_labels.max(), "Some classes are missing from the buffer"
        self.balance_scores = torch.gather(counts, 0, orig_inputs_idxs).float()

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            self.update_balancoir_scores(labels)
            self.merge_scores()
            index = np.random.choice(range(self.buffer_size), p=self.scores.cpu().numpy(), size=1)
            return index
        else:
            return -1


class ABSSampling(LARSSampling):
    def __init__(self, buffer_size: int, device: str, dataset: 'ContinualDataset'):
        super().__init__(buffer_size, device)
        self.dataset = dataset

    def scale_scores(self, past_indexes: torch.Tensor):
        # due normalizzazioni divere per i due gruppi
        past_importance = self.normalize_scores(self.importance_scores[past_indexes])
        current_importance = self.normalize_scores(self.importance_scores[~past_indexes])
        current_scores, past_scores = None, None
        if past_importance is not None:
            past_importance = 1 - past_importance
            past_scores = past_importance / past_importance.sum()
        if current_importance is not None:
            if current_importance.sum() == 0:
                current_importance += 1e-9
            current_scores = current_importance / current_importance.sum()

        return past_scores, current_scores

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        n_seen_classes, _ = self.dataset.get_offsets()

        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            past_indexes = labels < n_seen_classes

            past_scores, current_scores = self.scale_scores(past_indexes)
            past_percentage = np.float64(past_indexes.sum().cpu() / self.buffer_size)  # avoid numerical issues
            pres_percetage = 1 - past_percentage
            assert past_percentage + pres_percetage == 1, f"The sum of the percentages must be 1 but found {past_percentage+pres_percetage}: {past_percentage} + {pres_percetage}"
            rp = np.random.choice((0, 1), p=[past_percentage, pres_percetage])

            if not rp:
                index = np.random.choice(np.arange(self.buffer_size)[past_indexes.cpu().numpy()], p=past_scores.cpu().numpy(), size=1)
            else:
                index = np.random.choice(np.arange(self.buffer_size)[~past_indexes.cpu().numpy()], p=current_scores.cpu().numpy(), size=1)
            return index
        else:
            return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    buffer_size: int  # the maximum size of the buffer
    device: str  # the device to store the buffer on
    num_seen_examples: int  # the total number of examples seen, used for reservoir
    attributes: List[str]  # the attributes stored in the buffer
    attention_maps: List[torch.Tensor]  # (optional) attention maps used by TwF
    sample_selection_strategy: str  # the sample selection strategy used to select samples to replace. By default, 'reservoir'

    examples: torch.Tensor  # (mandatory) buffer attribute: the tensor of images
    labels: torch.Tensor  # (optional) buffer attribute: the tensor of labels
    logits: torch.Tensor  # (optional) buffer attribute: the tensor of logits
    task_labels: torch.Tensor  # (optional) buffer attribute: the tensor of task labels
    true_labels: torch.Tensor  # (optional) buffer attribute: the tensor of true labels

    def __init__(self, buffer_size: int, device="cpu", sample_selection_strategy='reservoir', **kwargs):
        """
        Initialize a reservoir-based Buffer object.

        Supports storing images, labels, logits, task_labels, and attention maps. This can be extended by adding more attributes to the `attributes` list and updating the `init_tensors` method accordingly.

        To select samples to replace, the buffer supports:
        - `reservoir` sampling: randomly selects samples to replace (default). Ref: "Jeffrey S Vitter. Random sampling with a reservoir."
        - `lars`: prioritizes retaining samples with the *higher* loss. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `labrs` (Loss-Aware Balanced Reservoir Sampling): combination of LARS and BRS. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `abs` (Asymmetric Balanced Sampling): for samples from the current task, prioritizes retaining samples with the *lower* loss (i.e., inverse `lossoir`); for samples from previous tasks, prioritizes retaining samples with the *higher* loss (i.e., `lossoir`). Useful for settings with noisy labels. Ref: "Monica Millunzi et al. May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels".

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str, optional): The device to store the buffer on. Defaults to "cpu".
            sample_selection_strategy: The sample selection strategy. Defaults to 'reservoir'. Options: 'reservoir', 'lars', 'labrs', 'abs', 'balancoir'.

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self._buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'true_labels']
        self.attention_maps = [None] * buffer_size
        self.sample_selection_strategy = sample_selection_strategy

        assert sample_selection_strategy.lower() in ['reservoir', 'lars', 'labrs', 'abs', 'balancoir', 'unlimited'], f"Invalid sample selection strategy: {sample_selection_strategy}"

        if sample_selection_strategy.lower() == 'abs':
            assert 'dataset' in kwargs, "The dataset is required for ABS sample selection"
            self.sample_selection_fn = ABSSampling(buffer_size, device, kwargs['dataset'])
        elif sample_selection_strategy.lower() == 'lars':
            self.sample_selection_fn = LARSSampling(buffer_size, device)
        elif sample_selection_strategy.lower() == 'labrs':
            self.sample_selection_fn = LossAwareBalancedSampling(buffer_size, device)
        elif sample_selection_strategy.lower() == 'unlimited':
            self.sample_selection_fn = lambda x: x
            self._buffer_size = 10  # initial buffer size, will be expanded if needed
        elif sample_selection_strategy.lower() == 'balancoir':
            self.sample_selection_fn = BalancoirSampling(buffer_size, device)
        else:
            self.sample_selection_fn = ReservoirSampling(buffer_size, device)

    def serialize(self, out_device='cpu'):
        """
        Serialize the buffer.

        Returns:
            A dictionary containing the buffer attributes.
        """
        return {attr_str: getattr(self, attr_str).to(out_device) for attr_str in self.attributes if hasattr(self, attr_str)}

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        self.sample_selection_fn.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        if self.sample_selection_strategy == 'unlimited':
            return self.num_seen_examples
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     true_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: tensor containing the true labels (used only for logging)
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):  # create tensor if not already present
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self._buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
            elif hasattr(self, attr_str):  # if tensor already exists, update it and possibly resize it according to the buffer_size
                if self.num_seen_examples < self._buffer_size:  # if the buffer is full, extend the tensor
                    old_tensor = getattr(self, attr_str)
                    pad = torch.zeros((self._buffer_size - old_tensor.shape[0], *attr.shape[1:]), dtype=old_tensor.dtype, device=self.device)
                    setattr(self, attr_str, torch.cat([old_tensor, pad], dim=0))

    @property
    def buffer_size(self):
        """
        Returns the buffer size.
        """
        if self.sample_selection_strategy == 'unlimited':
            # return max int if unlimited
            return int(1e9)
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        """
        Sets the buffer size.
        """
        if self.sample_selection_strategy != 'unlimited':
            self._buffer_size = value

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None, true_labels=None, sample_selection_scores=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            attention_maps: list of tensors containing the attention maps
            true_labels: if setting is noisy, the true labels associated with the examples. **Used only for logging.**
            sample_selection_scores: tensor containing the scores used for the sample selection strategy. NOTE: this is only used if the sample selection strategy defines the `update` method.

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, true_labels)

        for i in range(examples.shape[0]):
            if self.sample_selection_strategy == 'abs' or self.sample_selection_strategy == 'labrs':
                index = self.sample_selection_fn(self.num_seen_examples, labels=self.labels)
            elif self.sample_selection_strategy == 'balancoir':
                index = self.sample_selection_fn(self.num_seen_examples, labels=self.labels, proposed_class=labels[i])
            else:
                index = self.sample_selection_fn(self.num_seen_examples)
            self.num_seen_examples += 1
            if index >= 0:
                if self.sample_selection_strategy == 'unlimited' and self.num_seen_examples > self._buffer_size:
                    self._buffer_size *= 2
                    self.init_tensors(examples, labels, logits, task_labels, true_labels)
                if self.sample_selection_strategy == 'balancoir':
                    self.sample_selection_fn.update_unique_map(labels[i], self.labels[index] if index < self.num_seen_examples else None)

                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if attention_maps is not None:
                    self.attention_maps[index] = [at[i].byte().to(self.device) for at in attention_maps]
                if sample_selection_scores is not None:
                    self.sample_selection_fn.update(index, sample_selection_scores[i])
                if true_labels is not None:
                    self.true_labels[index] = true_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None,
                 mask_task_out=None, cpt=None, return_not_aug=False, not_aug_transform=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)
            return_not_aug: if True, also returns the not augmented items
            not_aug_transform: the transformation to be applied to the not augmented items (if `return_not_aug` is True)

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device = self.device if device is None else device

        if mask_task_out is not None:
            assert hasattr(self, 'task_labels') or cpt is not None
            assert hasattr(self, 'task_labels') or hasattr(self, 'labels')
            samples_mask = (self.task_labels != mask_task_out) if hasattr(self, 'task_labels') else self.labels // cpt != mask_task_out

        num_avail_samples = self.examples.shape[0] if mask_task_out is None else samples_mask.sum().item()
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)

        if size > min(num_avail_samples, self.examples.shape[0]):
            size = min(num_avail_samples, self.examples.shape[0])

        choice = np.random.choice(num_avail_samples, size=size, replace=False)
        if transform is None:
            def transform(x): return x

        selected_samples = self.examples[choice] if mask_task_out is None else self.examples[samples_mask][choice]

        if return_not_aug:
            if not_aug_transform is None:
                def not_aug_transform(x): return x
            ret_tuple = (apply_transform(selected_samples, transform=not_aug_transform).to(target_device),)
        else:
            ret_tuple = tuple()

        ret_tuple += (apply_transform(selected_samples, transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                selected_attr = attr[choice] if mask_task_out is None else attr[samples_mask][choice]
                ret_tuple += (selected_attr.to(target_device),)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

    def get_balanced_data(self, size: int, transform=None, n_classes=-1) -> Tuple:
        """
        Random samples a batch of size items only from n_classes, balancing the samples per class.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            n_classes: the number of classes to sample from

        Returns:
            a tuple containing the requested items.
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        tot_classes, class_counts = torch.unique(self.labels[:self.num_seen_examples], return_counts=True)
        if n_classes == -1:
            n_classes = len(tot_classes)

        finished = False
        selected = tot_classes
        while not finished:
            n_classes = min(n_classes, len(selected))
            size_per_class = torch.full([n_classes], size // n_classes)
            size_per_class[:size % n_classes] += 1
            selected = tot_classes[class_counts >= size_per_class[0]]
            if n_classes <= len(selected):
                finished = True
            if len(selected) == 0:
                print('WARNING: no class has enough examples')
                return self.get_data(size, transform=transform)

        selected = selected[torch.randperm(len(selected))[:n_classes]]

        choice = []
        for i, id_class in enumerate(selected):
            choice += np.random.choice(torch.where(self.labels[:self.num_seen_examples] == id_class)[0].cpu(),
                                       size=size_per_class[i].item(),
                                       replace=False).tolist()
        choice = np.array(choice)

        if transform is None:
            def transform(x): return x
        # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        ret_tuple = (apply_transform(self.examples[choice], transform=transform).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None, device=None) -> Tuple:
        """
        Returns the data by the given index.

        Args:
            index: the index of the item
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple containing the requested items. The returned items depend on the attributes stored in the buffer from previous calls to `add_data`.
        """
        target_device = self.device if device is None else device

        if transform is None:
            def transform(x): return x
        ret_tuple = (apply_transform(self.examples[indexes], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with all the items in the memory buffer
        """
        target_device = self.device if device is None else device
        if transform is None:
            ret_tuple = (self.examples[:len(self)].to(target_device),)
        else:
            ret_tuple = (apply_transform(self.examples[:len(self)], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)[:len(self)].to(target_device)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


@torch.no_grad()
def fill_buffer(buffer: Buffer, dataset: 'ContinualDataset', t_idx: int, net: 'MammothBackbone' = None, use_herding=False,
                required_attributes: List[str] = None, normalize_features=False, extend_equalize_buffer=False) -> None:
    """
    Adds examples from the current task to the memory buffer.
    Supports images, labels, task_labels, and logits.

    Args:
        buffer: the memory buffer
        dataset: the dataset from which take the examples
        t_idx: the task index
        net: (optional) the model instance. Used if logits are in buffer. If provided, adds logits.
        use_herding: (optional) if True, uses herding strategy. Otherwise, random sampling.
        required_attributes: (optional) the attributes to be added to the buffer. If None and buffer is empty, adds only examples and labels.
        normalize_features: (optional) if True, normalizes the features before adding them to the buffer
        extend_equalize_buffer: (optional) if True, extends the buffer to equalize the number of samples per class for all classes, even if that means exceeding the buffer size defined at initialization
    """
    if net is not None:
        mode = net.training
        net.eval()
    else:
        assert not use_herding, "Herding strategy requires a model instance"

    device = net.device if net is not None else get_device()

    n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])
    n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx])

    mask = dataset.train_loader.dataset.targets >= n_past_classes
    dataset.train_loader.dataset.targets = dataset.train_loader.dataset.targets[mask]
    dataset.train_loader.dataset.data = dataset.train_loader.dataset.data[mask]

    buffer.buffer_size = dataset.args.buffer_size  # reset initial buffer size

    if extend_equalize_buffer:
        samples_per_class = np.ceil(buffer.buffer_size / n_seen_classes).astype(int)
        new_bufsize = int(n_seen_classes * samples_per_class)
        if new_bufsize != buffer.buffer_size:
            print('Buffer size has bee changed to:', new_bufsize)
        buffer.buffer_size = new_bufsize
    else:
        samples_per_class = buffer.buffer_size // n_seen_classes

    # Check for requirs attributes
    required_attributes = required_attributes or ['examples', 'labels']
    assert all([attr in buffer.used_attributes for attr in required_attributes]) or len(buffer) == 0, \
        "Required attributes not in buffer: {}".format([attr for attr in required_attributes if attr not in buffer.used_attributes])

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_data = buffer.get_all_data()
        buf_y = buf_data[1]

        buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _buf_data_idx = {attr_name: _d[idx][:samples_per_class] for attr_name, _d in zip(required_attributes, buf_data)}
            buffer.add_data(**_buf_data_idx)

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x

    if 'logits' in buffer.used_attributes:
        assert net is not None, "Logits in buffer require a model instance"

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for data in loader:
        x, y, not_norm_x = data[0], data[1], data[2]
        if not x.size(0):
            continue
        a_x.append(not_norm_x.cpu())
        a_y.append(y.cpu())

        if net is not None:
            feats = net(norm_trans(not_norm_x.to(device)), returnt='features')
            outs = net.classifier(feats)
            if normalize_features:
                feats = feats / feats.norm(dim=1, keepdim=True)

            a_f.append(feats.cpu())
            a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y = torch.cat(a_x), torch.cat(a_y)
    if net is not None:
        a_f, a_l = torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y = a_x[idx], a_y[idx]

        if use_herding:
            _l = a_l[idx]
            feats = a_f[idx]
            mean_feat = feats.mean(0, keepdim=True)

            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

                idx_min = cost.argmin().item()

                buffer.add_data(
                    examples=_x[idx_min:idx_min + 1].to(device),
                    labels=_y[idx_min:idx_min + 1].to(device),
                    logits=_l[idx_min:idx_min + 1].to(device) if 'logits' in required_attributes else None,
                    task_labels=torch.ones(len(_x[idx_min:idx_min + 1])).to(device) * t_idx if 'task_labels' in required_attributes else None

                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1
        else:
            idx = torch.randperm(len(_x))[:samples_per_class]

            buffer.add_data(
                examples=_x[idx].to(device),
                labels=_y[idx].to(device),
                logits=_l[idx].to(device) if 'logits' in required_attributes else None,
                task_labels=torch.ones(len(_x[idx])).to(device) * t_idx if 'task_labels' in required_attributes else None
            )

    assert len(buffer.examples) <= buffer.buffer_size, f"buffer overflowed its maximum size: {len(buffer)} > {buffer.buffer_size}"
    assert buffer.num_seen_examples <= buffer.buffer_size, f"buffer has been overfilled, there is probably an error: {buffer.num_seen_examples} > {buffer.buffer_size}"

    if net is not None:
        net.train(mode)
