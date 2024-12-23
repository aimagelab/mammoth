"""
Custom buffer for Learning without Shortcuts (LwS).
"""
import torch
import numpy as np
from typing import Tuple
from torch.functional import Tensor
from torchvision import transforms
from torch.utils.data import Dataset

import math
from typing import Tuple

from utils.augmentations import apply_transform


class Buffer(Dataset):
    def __init__(self, buffer_size, device, n_tasks, attributes=['examples', 'labels', 'logits', 'task_labels'], n_bin=8):
        """
        Initializes the memory buffer.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the data
            n_tasks: the total number of tasks
            attributes: the attributes to store in the memory buffer
            n_bin: the number of bins for the reservoir binning strategy
        """

        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.task = 1
        self.task_number = n_tasks
        self.attributes = attributes
        self.delta = torch.zeros(buffer_size, device=device)

        self.balanced_class_perm = None
        self.num_bins = n_bin
        self.bins = np.zeros(self.num_bins)  # Initialize bins with zero counts
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.budget = (self.buffer_size // self.num_bins) // self.task
        self.num_examples = 0

    def reset_budget(self):
        self.task += 1
        self.budget = (self.buffer_size // self.num_bins) // self.task

    def reset_bins(self):
        self.bins = np.zeros(self.num_bins)
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.reset_budget()

    def update_loss_range(self, loss_value):
        """
        Updates the min and max loss values seen, for binning purposes.
        """
        self.min_loss = min(self.min_loss, loss_value)
        self.max_loss = max(self.max_loss, loss_value)

    def get_bin_index(self, loss_value):
        """
        Determines the bin index for a given loss value.
        """
        bin_range = self.max_loss - self.min_loss
        if bin_range == 0:
            return 0  # All losses are the same, only one bin needed
        bin_width = bin_range / self.num_bins
        bin_index = int((loss_value - self.min_loss) / bin_width)
        return min(bin_index, self.num_bins - 1)  # To handle the max loss

    def reservoir_bin_loss(self, loss_value: float) -> int:
        """
        Modified reservoir sampling algorithm considering loss values and binning.
        """
        self.update_loss_range(loss_value)
        bin_index = self.get_bin_index(loss_value)

        if self.bins[bin_index] < self.budget:
            if self.num_examples < self.buffer_size:
                self.bins[bin_index] += 1
                return self.num_examples
            else:
                rand = np.random.randint(0, self.buffer_size)
                self.bins[bin_index] += 1
                return rand
        else:
            return -1

    def reservoir_loss(self, num_seen_examples: int, buffer_size: int, loss_value: float) -> int:
        """
        Modified reservoir sampling algorithm considering loss values
        """
        # Probability based on the loss value (higher loss, higher probability)
        loss_probability = math.exp(loss_value) / (1 + math.exp(loss_value))
        rand = np.random.random()
        if rand < loss_probability and self.budget > 0:
            self.budget -= 1
            return np.random.randint(buffer_size)
        else:
            return -1

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))

        return self

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     clusters_labels=None, clusters_logits=None,
                     loss_values=None) -> None:
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                if attr_str.startswith('loss_val'):
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                         *attr.shape[1:]), dtype=typ, device=self.device) - 1)
                else:
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                         *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, clusters_labels=None, logits=None, clusters_logits=None, task_labels=None, loss_values=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, clusters_labels=clusters_labels, clusters_logits=clusters_logits, loss_values=loss_values)
        rix = []
        for i in range(examples.shape[0]):
            index = self.reservoir_bin_loss(loss_values[i])
            self.num_seen_examples += 1
            if index >= 0:
                self.num_examples += 1
                if self.examples.device != self.device:
                    self.examples.to(self.device)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.labels.device != self.device:
                        self.labels.to(self.device)
                    self.labels[index] = labels[i].to(self.device)
                if clusters_labels is not None:
                    if self.clusters_labels.device != self.device:
                        self.clusters_labels.to(self.device)
                    self.clusters_labels[index] = clusters_labels[i].to(self.device)
                if logits is not None:
                    if self.logits.device != self.device:
                        self.logits.to(self.device)
                    self.logits[index] = logits[i].to(self.device)
                if clusters_logits is not None:
                    if self.clusters_logits.device != self.device:
                        self.clusters_logits.to(self.device)
                    self.clusters_logits[index] = clusters_logits[i].to(self.device)
                if task_labels is not None:
                    if self.task_labels.device != self.device:
                        self.task_labels.to(self.device)
                    self.task_labels[index] = task_labels[i].to(self.device)
                if loss_values is not None:
                    if self.loss_values.device != self.device:
                        self.loss_values.to(self.device)
                    self.loss_values[index] = loss_values[i].to(self.device)

            rix.append(index)
        return torch.tensor(rix).to(self.device)

    def update_losses(self, loss_values, indexes):
        self.loss_values[indexes] = loss_values

    def get_losses(self):
        return self.loss_values.cpu().numpy()

    def get_task_labels(self):
        return self.task_labels.cpu().numpy()

    def get_data(self, size: int, transform: transforms = None, return_index=False, to_device=None) -> Tuple:
        m_t = min(self.num_examples, self.examples.shape[0])
        if size > m_t:
            size = m_t

        target_device = self.device if to_device is None else to_device

        choice = np.random.choice(m_t, size=size, replace=False)

        if transform is None:
            def transform(x): return x
        ret_tuple = (apply_transform(self.examples[choice], transform=transform).to(target_device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
