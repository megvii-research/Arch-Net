#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantize_helper import quant_weight, quant_feature


def get_out_scale(out_scale, out_channels):
    if isinstance(out_scale, (list, tuple)):
        out_scale = torch.from_numpy(np.array(out_scale).astype(np.float32))
    elif isinstance(out_scale, np.ndarray):
        out_scale = torch.from_numpy(out_scale.astype(np.float32))
    elif isinstance(out_scale, (int, float)):
        out_scale = torch.ones(out_channels) * out_scale
    else:
        raise AssertionError(
            "only support out_scale in ['int', 'float', 'list', 'tuple', 'numpy.ndarray']"
        )
    assert (
        len(out_scale) == out_channels
    ), "The length of out_scale does not match out_channels"
    assert out_scale.dim() == 1, "Multi-dimensional out_scale are not supported."
    return out_scale


class DorefaConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        weight_dtype,
        kernel_size=3,
        padding=(1, 1),
        stride=1,
        bias=True,
        quant_weight_mode="normal",
    ):
        super(DorefaConv2d, self).__init__()
        self.weight_dtype = weight_dtype
        self.padding = padding
        self.conv_Q = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )
        self.quant_weight_mode = quant_weight_mode
        self.stride = stride

    def forward(self, x):
        weight, _ = quant_weight(
            self.conv_Q.weight,
            self.weight_dtype,
            self.quant_weight_mode,
        )
        x = F.conv2d(
            x, weight, bias=self.conv_Q.bias, stride=self.stride, padding=self.padding
        )
        return x


class DorefaLinear1d(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        weight_dtype,
        bias=True,
        quant_weight_mode="normal",
    ):
        super(DorefaLinear1d, self).__init__()
        self.weight_dtype = weight_dtype
        self.linear_Q = nn.Linear(in_features, out_features, bias=bias)
        self.quant_weight_mode = quant_weight_mode

    def forward(self, x):
        weight, _ = quant_weight(
            self.linear_Q.weight,
            self.weight_dtype,
            self.quant_weight_mode,
        )
        x = F.linear(x, weight, bias=self.linear_Q.bias)
        return x


class DorefaQuantizeActivation(nn.Module):
    def __init__(
        self,
        feature_dtype,
        out_channels,
        out_scale=1.0,
        multiplier=None,  # magic number
        after_bn=True,  # if there is a bn layer before this layer, we use a magic number
    ):
        super(DorefaQuantizeActivation, self).__init__()
        self.feature_dtype = feature_dtype
        self.out_scale = get_out_scale(out_scale, out_channels)
        if multiplier is None:
            multiplier = 0.1 if after_bn else 1.0
        self.mul = multiplier

    def forward(self, x):
        x = x * self.mul
        x = quant_feature(x, self.feature_dtype, self.out_scale.to(x.device))
        return x
