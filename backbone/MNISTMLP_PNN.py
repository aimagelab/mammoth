# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone, num_flat_features, xavier
from backbone.utils.modules import AlphaModule, ListModule


class MNISTMLP_PNN(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset, equipped with lateral connection.
    """

    def __init__(self, input_size: int, output_size: int,
                 old_cols: List[AlphaModule] = None) -> None:
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
            old_cols: a list of all the old columns
        """
        super(MNISTMLP_PNN, self).__init__()

        if old_cols is None:
            old_cols = []

        self.old_cols = []

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.classifier = nn.Linear(100, self.output_size)
        if len(old_cols) > 0:
            self.old_fc1s = ListModule()
            self.old_fc2s = ListModule()
            self.base_1 = nn.Sequential(
                nn.Linear(100 * len(old_cols), 100),
                nn.ReLU(),
                nn.Linear(100, 100, bias=False)
            )
            self.base_2 = nn.Sequential(
                nn.Linear(100 * len(old_cols), 100),
                nn.ReLU(),
                nn.Linear(100, self.output_size, bias=False)
            )

            self.adaptor1 = nn.Sequential(AlphaModule(100 * len(old_cols)),
                                          self.base_1)
            self.adaptor2 = nn.Sequential(AlphaModule(100 * len(old_cols)),
                                          self.base_2)

            for old_col in old_cols:
                self.old_fc1s.append(
                    nn.Sequential(nn.Linear(self.input_size, 100), nn.ReLU()))
                self.old_fc2s.append(
                    nn.Sequential(nn.Linear(100, 100), nn.ReLU()))
                self.old_fc1s[-1][0].load_state_dict(old_col.fc1.state_dict())
                self.old_fc2s[-1][0].load_state_dict(old_col.fc2.state_dict())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.fc1.apply(xavier)
        self.fc2.apply(xavier)
        self.classifier.apply(xavier)
        if len(self.old_cols) > 0:
            self.adaptor1.apply(xavier)
            self.adaptor2.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Retruns:
            output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))
        if len(self.old_cols) > 0:
            with torch.no_grad():
                fc1_kb = [old(x) for old in self.old_fc1s]
                fc2_kb = [old(fc1_kb[i]) for i, old in enumerate(self.old_fc2s)]
            x = F.relu(self.fc1(x))

            y = self.adaptor1(torch.cat(fc1_kb, 1))
            x = F.relu(self.fc2(x) + y)

            y = self.adaptor2(torch.cat(fc2_kb, 1))
            x = self.classifier(x) + y
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.classifier(x)
        if returnt == 'out':
            return x

        raise NotImplementedError("Unknown return type")
