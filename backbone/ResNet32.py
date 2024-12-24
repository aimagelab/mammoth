import math
from torch import nn
from torch.nn import functional as F

from backbone import MammothBackbone, register_backbone, xavier


EPS_BATCH_NORM = 1e-4


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IdentityShortcut(nn.Module):
    def __init__(self, planes):
        super(IdentityShortcut, self).__init__()
        self.planes = planes
        self.pad_dimension = planes // 4

    def forward(self, x):
        x = x[:, :, ::2, ::2]
        # add padding to the channels dimension to match the output of the residual
        return F.pad(x, (0, 0, 0, 0, self.pad_dimension, self.pad_dimension))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.last = last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=EPS_BATCH_NORM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=EPS_BATCH_NORM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        # out = self.relu(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if not self.last:
            out = self.relu(out)
        return out


class ResNet32(MammothBackbone):

    def __init__(self, depth=32, num_classes=1000):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        n = (depth - 2) // 6

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(16, eps=EPS_BATCH_NORM)
        self.layer1 = self._make_layer(BasicBlock, 16, n)
        self.layer2 = self._make_layer(BasicBlock, 32, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, n, stride=2, last=True)
        self.bn_final = nn.BatchNorm2d(64 * BasicBlock.expansion, eps=EPS_BATCH_NORM)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * BasicBlock.expansion, num_classes)

        gain = math.sqrt(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels  # m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # theano init HeNormal with gain='relu' (from Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing human-level performance on imagenet classification)
                m.weight.data.normal_(0, gain * math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.classifier.weight.data.normal_(0, math.sqrt(1. / 64))
        self.classifier.bias.data.zero_()
        # self.classifier.apply(xavier)

    def _make_layer(self, block, planes, blocks, stride=1, last=False):
        downsample = None
        if stride != 1:
            downsample = IdentityShortcut(planes)
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            # )
        if last:
            blocks -= 1

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        if last:
            layers.append(block(self.inplanes, planes, last=True))
        return nn.Sequential(*layers)

    def forward(self, x, returnt='out'):
        x = self.bn_initial(self.relu(self.conv1(x)))

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        features = x.view(x.size(0), -1)

        if returnt == 'features':
            return features

        out = self.classifier(features)
        if returnt == 'both':
            return (out, features)
        return out


@register_backbone('resnet32')
def resnet32(num_classes: int, depth: int = 32):
    """
    Constructs a ResNet-32 model, as used in 'iCaRL'.
    """
    return ResNet32(num_classes=num_classes, depth=depth)
