#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

from einops.layers.torch import Rearrange

class ARoLLayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class Linear(nn.Linear, ARoLLayer):

    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            ctx_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ARoLLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.ctx_features = ctx_features

        # Actual trainable parameters
        if r > 0:

            self.lora_A = nn.Sequential(
                Rearrange("B C -> B 1 C"),
                nn.Conv1d(1, r, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(r, r, kernel_size=1),
                nn.LayerNorm(in_features)
            )

            self.lora_B = nn.Sequential(
                Rearrange("B C -> B 1 C"),
                nn.Conv1d(1, r, kernel_size=1),
                nn.GELU(),
                Rearrange("B N C -> B C N"),
                nn.Conv1d(ctx_features, out_features, kernel_size=1, groups=8),
                nn.LayerNorm(r)
            )

            #self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            #self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.r > 0:
            lA = self.lora_A(ctx).transpose(1, 2)
            lB = self.lora_B(ctx).transpose(1, 2)

            result += (self.lora_dropout(x) @ lA @ lB) * self.scaling

        return result


