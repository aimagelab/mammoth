# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, num_flat_features, xavier


class MNISTMLP(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
        """
        super(MNISTMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))

        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'full':
            return (out, feats)

        raise NotImplementedError("Unknown return type")

    def to(self, device):
        super().to(device)
        self.device = device
        return self
