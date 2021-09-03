'''=================================================

@Project -> File：ST-Transformer->base_layers

@IDE：PyCharm

@coding: utf-8

@time:2021/7/23 15:27

@author:Pengzhangzhi

@Desc：
=================================================='''
import torch.nn as nn
import torch
from typing import Type, Any, Callable, Union, List, Optional

from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = conv1x1(inplanes, planes)
        self.convback = conv1x1(planes, inplanes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)  # inplanes -> planes
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # planes -> planes
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # inplanes -> planes

        out += identity  # planes

        out = self.convback(out)
        out = self.relu(out)

        return out


if __name__ == '__main__':



    pass

