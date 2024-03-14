"""
This module contains a version of the reservoir buffer that is specifically designed for the GSS model.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, minibatch_size, model=None):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels']
        self.model = model
        self.minibatch_size = minibatch_size
        self.cache = {}
        self.fathom = 0
        self.fathom_mask = None
        self.reset_fathom()

        self.conterone = 0

    def reset_fathom(self):
        self.fathom = 0
        self.fathom_mask = torch.randperm(min(self.num_seen_examples, self.examples.shape[0] if hasattr(self, 'examples') else self.num_seen_examples))

    def get_grad_score(self, batch_x, batch_y, X, Y, indices):
        g = self.model.get_grads(batch_x, batch_y)
        G = []
        for x, y, idx in zip(X, Y, indices):
            if idx in self.cache:
                grd = self.cache[idx]
            else:
                grd = self.model.get_grads(x.unsqueeze(0), y.unsqueeze(0))
                self.cache[idx] = grd
            G.append(grd)
        G = torch.cat(G).to(g.device)
        c_score = 0
        grads_at_a_time = 5
        # let's split this so your gpu does not melt. You're welcome.
        for it in range(int(np.ceil(G.shape[0] / grads_at_a_time))):
            tmp = F.cosine_similarity(g, G[it * grads_at_a_time: (it + 1) * grads_at_a_time], dim=1).max().item() + 1
            c_score = max(c_score, tmp)
        return c_score

    def functional_reservoir(self, x, y, batch_c, bigX=None, bigY=None, indices=None):
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples, batch_c

        elif batch_c < 1:
            single_c = self.get_grad_score(x.unsqueeze(0), y.unsqueeze(0), bigX, bigY, indices)
            s = self.scores.cpu().numpy()
            i = np.random.choice(np.arange(0, self.buffer_size), size=1, p=s / s.sum())[0]
            rand = np.random.rand(1)[0]
            # print(rand, s[i] / (s[i] + c))
            if rand < s[i] / (s[i] + single_c):
                return i, single_c

        return -1, 0

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor) -> None:
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
        self.scores = torch.zeros((self.buffer_size, *attr.shape[1:]),
                                  dtype=torch.float32, device=self.device)

    def add_data(self, examples, labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels)

        # compute buffer score
        if self.num_seen_examples > 0:
            bigX, bigY, indices = self.get_data(min(self.minibatch_size, self.num_seen_examples), give_index=True,
                                                random=True)
            c = self.get_grad_score(examples, labels, bigX, bigY, indices)
        else:
            bigX, bigY, indices = None, None, None
            c = 0.1

        for i in range(examples.shape[0]):
            index, score = self.functional_reservoir(examples[i], labels[i], c, bigX, bigY, indices)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                self.scores[index] = score
                if index in self.cache:
                    del self.cache[index]

    def drop_cache(self):
        self.cache = {}

    def get_data(self, size: int, transform: transforms = None, give_index=False, random=False) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with the requested items
        """

        if size > self.examples.shape[0]:
            size = self.examples.shape[0]

        if random:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                      size=min(size, self.num_seen_examples),
                                      replace=False)
        else:
            choice = np.arange(self.fathom, min(self.fathom + size, self.examples.shape[0], self.num_seen_examples))
            choice = self.fathom_mask[choice]
            self.fathom += len(choice)
            if self.fathom >= self.examples.shape[0] or self.fathom >= self.num_seen_examples:
                self.fathom = 0
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if give_index:
            ret_tuple += (choice,)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
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
