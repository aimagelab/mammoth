from argparse import Namespace
import importlib
import inspect
import os
import math

import numpy as np
import torch
import torch.nn as nn

from typing import Callable

from utils import register_dynamic_module_fn

REGISTERED_BACKBONES = dict()  # dictionary containing the registered networks. Template: {name: {'class': class, 'parsable_args': parsable_args}}


def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    Args:
        m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def num_flat_features(x: torch.Tensor) -> int:
    """
    Computes the total number of items except the first (batch) dimension.

    Args:
        x: input tensor

    Returns:
        number of item from the second dimension onward
    """
    size = x.size()[1:]
    num_features = 1
    for ff in size:
        num_features *= ff
    return num_features


class MammothBackbone(nn.Module):
    """
    A backbone module for the Mammoth model.

    Args:
        **kwargs: additional keyword arguments

    Methods:
        forward: Compute a forward pass.
        features: Get the features of the input tensor (same as forward but with returnt='features').
        get_params: Returns all the parameters concatenated in a single tensor.
        set_params: Sets the parameters to a given value.
        get_grads: Returns all the gradients concatenated in a single tensor.
        get_grads_list: Returns a list containing the gradients (a tensor for each layer).
    """

    def __init__(self, **kwargs) -> None:
        super(MammothBackbone, self).__init__()
        self.device = torch.device('cpu') if 'device' not in kwargs else kwargs['device']

    def to(self, device, *args, **kwargs):
        super(MammothBackbone, self).to(device, *args, **kwargs)
        self.device = device
        return self

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among `out`, `features`, `both`, or `all`)

        Returns:
            output tensor
        """
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the features of the input tensor.

        Args:
            x: input tensor

        Returns:
            features tensor
        """
        return self.forward(x, returnt='features')

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.

        Returns:
            parameters tensor
        """
        return torch.nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.

        Args:
            new_params: concatenated values to be set
        """
        torch.nn.utils.vector_to_parameters(new_params, self.parameters())

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def set_grads(self, new_grads: torch.Tensor) -> None:
        """
        Sets the gradients of all parameters.

        Args:
            new_params: concatenated values to be set
        """
        progress = 0
        for pp in list(self.parameters()):
            cand_grads = new_grads[progress: progress +
                                   torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.grad = cand_grads


def register_backbone(name: str) -> Callable:
    """
    Decorator to register a backbone network for use in a Dataset. The decorator may be used on a class that inherits from `MammothBackbone` or on a function that returns a `MammothBackbone` instance.
    The registered model can be accessed using the `get_backbone` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the backbone network's class. The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the backbone network
    """

    return register_dynamic_module_fn(name, REGISTERED_BACKBONES, MammothBackbone)


def get_backbone_class(name: str, return_args=False) -> MammothBackbone:
    """
    Get the backbone network class from the registered networks.

    Args:
        name: the name of the backbone network
        return_args: whether to return the parsable arguments of the backbone network

    Returns:
        the backbone class
    """
    name = name.replace('_', '-').lower()
    assert name in REGISTERED_BACKBONES, f"Attempted to access non-registered network: {name}"
    cl = REGISTERED_BACKBONES[name]['class']
    if return_args:
        return cl, REGISTERED_BACKBONES[name]['parsable_args']


def get_backbone(args: Namespace) -> MammothBackbone:
    """
    Build the backbone network from the registered networks.

    Args:
        args: the arguments which contains the `--backbone` attribute and the additional arguments required by the backbone network

    Returns:
        the backbone model
    """
    backbone_class, backbone_args = get_backbone_class(args.backbone, return_args=True)
    missing_args = [arg for arg in backbone_args.keys() if arg not in vars(args)]
    assert len(missing_args) == 0, "Missing arguments for the backbone network: " + ', '.join(missing_args)

    parsed_args = {arg: getattr(args, arg) for arg in backbone_args.keys()}

    return backbone_class(**parsed_args)


# import all files in the backbone folder to register the networks
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file != '__init__.py':
        importlib.import_module(f'backbone.{file[:-3]}')
