'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''
import math
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import ops


def log1mexp(x, expm1_guard=1e-7):
    # See https://cran.r-project.org/package=Rmpfr/.../log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[~t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1 + expm1_guard).log()  # limits magnitude of gradient

    y[~t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y


class NeuralNearestNeighbors(nn.Module):
    r"""
    Computes neural nearest neighbor volumes based on pairwise distances
    """

    def __init__(self, k, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: temperature options:
            external_temp: Whether temperature is given as external input
                rather than fixed parameter
            temp_bias: A fixed bias to add to the log temperature
            distance_bn: Whether to put distances through a batchnorm layer
        """
        super(NeuralNearestNeighbors, self).__init__()
        self.external_temp = temp_opt.get("external_temp")
        self.log_temp_bias = log(temp_opt.get("temp_bias", 1))
        distance_bn = temp_opt.get("distance_bn")

        if not self.external_temp:
            self.log_temp = nn.Parameter(torch.FloatTensor(1).fill_(0.0))
        if distance_bn:
            self.bn = nn.BatchNorm1d(1)
        else:
            self.bn = None

        self.k = k

    def forward(self, D, log_temp=None):
        b, m, o = D.shape
        if self.bn is not None:
            D = self.bn(D.view(b, 1, m * o)).view(D.shape)

        if self.external_temp:
            log_temp = log_temp.view(D.shape[0], D.shape[1], -1)
        else:
            log_temp = self.log_temp.view(1, 1, 1)

        log_temp = log_temp + self.log_temp_bias

        temperature = log_temp.exp()
        if self.training:
            M = D.data > -float("Inf")
            if len(temperature) > 1:
                D[M] /= temperature.expand_as(D)[M]
            else:
                D[M] = D[M] / temperature[0, 0, 0]
        else:
            D /= temperature

        logits = D.view(D.shape[0] * D.shape[1], -1)

        samples_arr = []

        for r in range(self.k):
            # Eqs. 8 and 10
            weights = F.log_softmax(logits, dim=1)
            # weights_exp = ops.clamp_probs(weights.exp())
            weights_exp = weights.exp()

            samples_arr.append(weights_exp.view(b, m, o))

            # Eq. 9
            logits = logits + log1mexp(weights.view(*logits.shape))
            # logits = logits + (1-weights_exp.view(*logits.shape)).log()

        W = torch.stack(samples_arr, dim=3)

        return W
