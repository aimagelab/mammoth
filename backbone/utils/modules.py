# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AlphaModule(nn.Module):
    def __init__(self, shape):
        super(AlphaModule, self).__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.alpha = Parameter(torch.rand(tuple([1] + list(shape))) * 0.1,
                               requires_grad=True)

    def forward(self, x):
        return x * self.alpha

    def parameters(self, recurse: bool = True):
        yield self.alpha


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.idx = 0
        for module in args:
            self.add_module(str(self.idx), module)
            self.idx += 1

    def append(self, module):
        self.add_module(str(self.idx), module)
        self.idx += 1

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.idx
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
