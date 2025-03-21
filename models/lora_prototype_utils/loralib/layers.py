#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer():
    def __init__(
            self,
            lora_dropout: float,
    ):
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class Linear(nn.Linear, LoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor, AB: dict = None):

        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            B = AB['B']
            A = AB.get('A')
            if A is not None:
                return result + (B @ (A @ x.transpose(1, 2).unsqueeze(1))).sum(1).transpose(1, 2)
            return result + (B @ x.transpose(1, 2).unsqueeze(1)).sum(1).transpose(1, 2)

        return result


class ClipLinear(nn.Linear, LoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor, AB: dict = None):

        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            B = AB['B']
            A = AB.get('A')
            if A is not None:
                res = (B @ (A @ torch.permute(x, (1, 2, 0)).unsqueeze(1))).sum(1)
                return result + torch.permute(res, (2, 0, 1))
            res = (B @ torch.permute(x, (1, 2, 0)).unsqueeze(1)).sum(1)
            return result + torch.permute(res, (2, 0, 1))

        return result