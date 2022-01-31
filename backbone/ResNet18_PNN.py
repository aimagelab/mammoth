# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from backbone.utils.modules import ListModule, AlphaModule
from backbone.ResNet18 import conv3x3
from backbone.ResNet18 import BasicBlock
from backbone.ResNet18 import ResNet
from typing import List


class BasicBlockPnn(BasicBlock):
    """
    The basic block of ResNet. Modified for PNN.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ResNetPNN(ResNet):
    """
    ResNet network architecture modified for PNN.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, old_cols: List[nn.Module]=None,
                 x_shape: torch.Size=None):
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNetPNN, self).__init__(block, num_blocks, num_classes, nf)
        if old_cols is None:
            old_cols = []

        self.old_cols = old_cols
        self.x_shape = x_shape
        self.classifier = self.linear
        if len(old_cols) == 0:
            return

        assert self.x_shape is not None
        self.in_planes = self.nf
        self.lateral_classifier = nn.Linear(nf * 8, num_classes)
        self.adaptor4 = nn.Sequential(
            AlphaModule((nf * 8 * len(old_cols), 1, 1)),
            nn.Conv2d(nf * 8 * len(old_cols), nf * 8, 1),
            nn.ReLU()
        )
        for i in range(5):
            setattr(self, 'old_layer' + str(i) + 's', ListModule())

        for i in range(1, 4):
            factor = 2 ** (i-1)
            setattr(self, 'lateral_layer' + str(i + 1),
                self._make_layer(block, nf * (2 ** i), num_blocks[i], stride=2)
            )
            setattr(self, 'adaptor' + str(i),
                nn.Sequential(
                    AlphaModule((nf * len(old_cols) * factor,
                        self.x_shape[2] // factor, self.x_shape[3] // factor)),
                    nn.Conv2d(nf * len(old_cols) * factor, nf * factor, 1),
                    nn.ReLU(),
                    getattr(self, 'lateral_layer' + str(i+1))
            ))
        for old_col in old_cols:
            self.in_planes = self.nf
            self.old_layer0s.append(conv3x3(3, nf * 1))
            self.old_layer0s[-1].load_state_dict(old_col.conv1.state_dict())
            for i in range(1, 5):
                factor = (2 ** (i - 1))
                layer = getattr(self, 'old_layer' + str(i) + 's')
                layer.append(self._make_layer(block, nf * factor,
                             num_blocks[i-1], stride=(1 if i == 1 else 2)))
                old_layer = getattr(old_col, 'layer' + str(i))
                layer[-1].load_state_dict(old_layer.state_dict())

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(nn.ReLU())
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        layers.append(nn.ReLU())
        return nn.Sequential(*(layers[1:]))

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        if self.x_shape is None:
            self.x_shape = x.shape
        if len(self.old_cols) == 0:
            return super(ResNetPNN, self).forward(x)
        else:
            with torch.no_grad():
                out0_old = [relu(self.bn1(old(x))) for old in self.old_layer0s]
                out1_old = [old(out0_old[i]) for i, old in enumerate(self.old_layer1s)]
                out2_old = [old(out1_old[i]) for i, old in enumerate(self.old_layer2s)]
                out3_old = [old(out2_old[i]) for i, old in enumerate(self.old_layer3s)]
                out4_old = [old(out3_old[i]) for i, old in enumerate(self.old_layer4s)]

            out = relu(self.bn1(self.conv1(x)))
            out = F.relu(self.layer1(out))
            y = self.adaptor1(torch.cat(out1_old, 1))
            out = F.relu(self.layer2(out) + y)
            y = self.adaptor2(torch.cat(out2_old, 1))
            out = F.relu(self.layer3(out) + y)
            y = self.adaptor3(torch.cat(out3_old, 1))
            out = F.relu(self.layer4(out) + y)
            out = avg_pool2d(out, out.shape[2])
            out = out.view(out.size(0), -1)

            y = avg_pool2d(torch.cat(out4_old, 1), out4_old[0].shape[2])
            y = self.adaptor4(y)
            y = y.view(out.size(0), -1)
            y = self.lateral_classifier(y)
            out = self.linear(out) + y
        
        if returnt == 'out':
            return out

        raise NotImplementedError("Unknown return type")


def resnet18_pnn(nclasses: int, nf: int=64,
                 old_cols: List[nn.Module]=None, x_shape: torch.Size=None):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    if old_cols is None:
        old_cols = []
    return ResNetPNN(BasicBlockPnn, [2, 2, 2, 2], nclasses, nf,
                     old_cols=old_cols, x_shape=x_shape)
