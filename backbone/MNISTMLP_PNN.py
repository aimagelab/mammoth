# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import xavier, num_flat_features
from backbone.utils.modules import ListModule, AlphaModule
from typing import List


class MNISTMLP_PNN(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset, equipped with lateral connection.
    """

    def __init__(self, input_size: int, output_size: int,
                 old_cols: List[AlphaModule] = None) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        :param old_cols: a list of all the old columns
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
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
        return x

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (input_size * 100
                    + 100 + 100 * 100 + 100 + 100 * output_size + output_size)
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
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads
