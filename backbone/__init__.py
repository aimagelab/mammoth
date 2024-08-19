from argparse import Namespace
import importlib
import inspect
import os
import math

import torch
import torch.nn as nn

from typing import Callable

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
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.

        Args:
            new_params: concatenated values to be set
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                     torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).

        Returns:
            gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads


def register_backbone(name: str) -> Callable:
    """
    Decorator to register a backbone network for use in a Dataset. The decorator may be used on a class that inherits from `MammothBackbone` or on a function that returns a `MammothBackbone` instance.
    The registered model can be accessed using the `get_backbone` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the backbone network's class. The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the backbone network
    """
    def register_network_fn(cls: MammothBackbone | Callable) -> MammothBackbone:
        # check if the name is already registered
        if name in REGISTERED_BACKBONES:
            raise ValueError(f"Name {name} already registered!")

        # check if `cls` is a subclass of `MammothBackbone`
        if inspect.isfunction(cls):
            signature = inspect.signature(cls)
        elif isinstance(cls, MammothBackbone) or issubclass(cls, MammothBackbone):
            signature = inspect.signature(cls.__init__)
        else:
            raise ValueError("The registered class must be a subclass of MammothBackbone or a function returning MammothBackbone")

        parsable_args = {}
        for arg_name, value in list(signature.parameters.items()):
            if arg_name != 'self' and not arg_name.startswith('_'):
                default = value.default
                tp = str if default is inspect.Parameter.empty or value.annotation is inspect._empty else type(default)
                if default is inspect.Parameter.empty and arg_name != 'num_classes':
                    parsable_args[arg_name] = {
                        'type': tp,
                        'required': True
                    }
                else:
                    parsable_args[arg_name] = {
                        'type': tp,
                        'required': False,
                        'default': default if default is not inspect.Parameter.empty else None
                    }

        REGISTERED_BACKBONES[name] = {'class': cls, 'parsable_args': parsable_args}
        return cls

    return register_network_fn


def get_backbone_class(name: str, return_args=False) -> MammothBackbone:
    assert name in REGISTERED_BACKBONES, "Attempted to access non-registered network"
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


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name, _ = os.path.splitext(file)
        relative_module_name = f".{module_name}"
        module = importlib.import_module(relative_module_name, package=__name__)
