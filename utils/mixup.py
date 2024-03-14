# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributions.beta import Beta


def mixup(couples, alpha, force_lambda=None):
    """
    Applies mixup augmentation to the given couples of inputs.

    Args:
        couples (list): A list of tuples, where each tuple contains two inputs to be mixed.
        alpha (float): The alpha parameter for the Beta distribution used to sample the mixing coefficients.
        force_lambda (float or None, optional): If not None, forces the use of a specific mixing coefficient for all inputs.

    Returns:
        tuple or torch.Tensor: If more than one mixed input is generated, a tuple of mixed inputs is returned. Otherwise, a single mixed input is returned.
    """
    lamda = Beta(alpha, alpha).rsample((len(couples[0][0]),)).to(couples[0][0].device)
    lamda = torch.max(lamda, 1 - lamda)

    if force_lambda is not None:
        lamda = torch.tensor(force_lambda).repeat((len(couples[0][0]),)).to(couples[0][0].device)

    returns = []

    for (i1, i2) in couples:
        lamda = lamda.view([lamda.shape[0]] + [1] * (len(i1.shape) - 1))
        assert i1.shape == i2.shape
        x_out = lamda * i1 + (1 - lamda) * i2
        returns.append(x_out)

    return tuple(returns) if len(returns) > 1 else returns[0]
