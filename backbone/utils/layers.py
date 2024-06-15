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


class LoRALinear(nn.Linear, LoRALayer):

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
            A = None
            if isinstance(AB, dict):
                B = AB['B']
                A = AB.get('A')
            else:
                B = AB
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
            A = None
            if isinstance(AB, dict):
                B = AB['B']
                A = AB.get('A')
            else:
                B = AB
            if A is not None:
                res = (B @ (A @ torch.permute(x, (1, 2, 0)).unsqueeze(1))).sum(1)
                return result + torch.permute(res, (2, 0, 1))
            res = (B @ torch.permute(x, (1, 2, 0)).unsqueeze(1)).sum(1)
            return result + torch.permute(res, (2, 0, 1))

        return result


class IncrementalClassifier(nn.Module):

    def __init__(self, embed_dim: int, nb_classes: int):
        """
        Incremental classifier for continual learning.

        Args:
            embed_dim: int, dimension of the input features.
            nb_classes: int, number of classes to classify.
        """

        super().__init__()

        self.embed_dim = embed_dim

        heads = [nn.Linear(embed_dim, nb_classes, bias=True)]

        self.heads = nn.ModuleList(heads)
        self.old_state_dict = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update(self, nb_classes: int, freeze_old=True):
        """
        Add a new head to the classifier.

        Args:
            nb_classes, number of classes to add.
            freeze_old: bool, whether to freeze the old heads.
        """

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True).to(self.get_device())

        nn.init.trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)

        if freeze_old:
            for param in self.heads.parameters():
                param.requires_grad = False

        self.heads.append(_fc)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Compute the logits for each head and concatenate them.

        Args:
            x: torch.Tensor, input features.
        """
        return torch.cat([h(x) for h in self.heads], dim=1)
