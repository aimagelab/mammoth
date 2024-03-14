# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.augmentations import apply_transform
from utils.conf import get_device


def icarl_replay(self: ContinualModel, dataset, val_set_split=0):
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

            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True,
                                                          num_workers=self.args.num_workers)


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.

    Args:
        num_seen_examples: the number of seen examples
        buffer_size: the maximum buffer size

    Returns:
        the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device="cpu"):
        """
        Initialize a reservoir-based Buffer object.

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str, optional): The device to store the buffer on. Defaults to "cpu".

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.attention_maps = [None] * buffer_size

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if attention_maps is not None:
                    self.attention_maps[index] = [at[i].byte().to(self.device) for at in attention_maps]

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None, mask_task_out=None, cpt=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)

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
        ret_tuple = (torch.stack([transform(ee) for ee in selected_samples.cpu()]).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                selected_attr = attr[choice] if mask_task_out is None else attr[samples_mask][choice]
                ret_tuple += (selected_attr.to(target_device),)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

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
        ret_tuple = (apply_transform(self.examples[:len(self)], transform=transform).to(target_device),)
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
            def transform(x): return x

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
def fill_buffer(buffer: Buffer, dataset: ContinualDataset, t_idx: int, net: ContinualModel = None, use_herding=False, required_attributes: List[str] = None) -> None:
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
    for x, y, not_norm_x in loader:
        mask = (y >= n_past_classes) & (y < n_seen_classes)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        a_x.append(not_norm_x.cpu())
        a_y.append(y.cpu())

        if net is not None:
            feats = net(norm_trans(not_norm_x.to(device)), returnt='features')
            outs = net.classifier(feats)
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

    assert len(buffer.examples) <= buffer.buffer_size
    assert buffer.num_seen_examples <= buffer.buffer_size

    if net is not None:
        net.train(mode)
