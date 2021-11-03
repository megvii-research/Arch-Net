#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from operators.dorefa.dorefa import DorefaConv2d, DorefaLinear1d, DorefaQuantizeActivation
from operators.dorefa.data_type import DATA_TYPE_DICT
import sys
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SplitDownsample(nn.Module):
    def __init__(self, block, planes, stride, weight_dtype, feature_dtype):
        super(SplitDownsample, self).__init__()
        self.max_ic = 512
        self.conv1 = DorefaConv2d(self.max_ic, planes*block.expansion//2, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes*block.expansion//2)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*block.expansion//2)
        self.conv2 = DorefaConv2d(self.max_ic, planes*block.expansion//2, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes*block.expansion//2)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*block.expansion//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x[:, :self.max_ic])
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.quantize_activation_1(x1)
        x2 = self.conv2(x[:, self.max_ic:])
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.quantize_activation_2(x2)

        return torch.cat((x1, x2), dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
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
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class QuantizeBottleneckV1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = DorefaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv = DorefaConv2d(planes*self.expansion*2, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn = nn.BatchNorm2d(planes*self.expansion)
        self.concat_quantize_activation = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quantize_activation_1(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.fa_res:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.cat((out, identity), dim=1)
        out = self.concat_conv(out)
        out = self.concat_bn(out)
        out = self.relu(out)
        out = self.concat_quantize_activation(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.fa_res:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class QuantizeBottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_max_ic = 256
        self.conv1 = DorefaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv_1 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_1 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_2 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_2 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quantize_activation_1(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.fa_res:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        identity1 = identity[:, :self.half_max_ic]
        identity2 = identity[:, self.half_max_ic:]

        out1 = out[:, :self.half_max_ic]
        out2 = out[:, self.half_max_ic:]

        out1 = self.concat_conv_1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn_1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv_2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn_2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out2 = torch.cat((out1, out2), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.fa_res:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class QuantizeBottleneckV3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_max_ic = 256
        self.conv1 = DorefaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv_1 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_1 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_2 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_2 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_3 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_3 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_4 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_4 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_4 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quantize_activation_1(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.is_train:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        identity1 = identity[:, :self.half_max_ic]
        identity2 = identity[:, self.half_max_ic:self.half_max_ic*2]
        identity3 = identity[:, self.half_max_ic*2:self.half_max_ic*3]
        identity4 = identity[:, self.half_max_ic*3:]

        out1 = out[:, :self.half_max_ic]
        out2 = out[:, self.half_max_ic:self.half_max_ic*2]
        out3 = out[:, self.half_max_ic*2:self.half_max_ic*3]
        out4 = out[:, self.half_max_ic*3:]

        out1 = self.concat_conv_1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn_1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv_2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn_2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out3 = self.concat_conv_3(torch.cat((out3, identity3), dim=1))
        out3 = self.concat_bn_3(out3)
        out3 = self.relu(out3)
        out3 = self.concat_quantize_activation_3(out3)
        out4 = self.concat_conv_4(torch.cat((out4, identity4), dim=1))
        out4 = self.concat_bn_4(out4)
        out4 = self.relu(out4)
        out4 = self.concat_quantize_activation_4(out4)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.fa_res:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class QuantizeBottleneckV4(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_max_ic = 256
        self.conv1_1 = DorefaConv2d(inplanes//2, planes//2, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_1 = nn.BatchNorm2d(planes//2)
        self.quantize_activation_1_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//2)
        self.conv1_2 = DorefaConv2d(inplanes//2, planes//2, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_2 = nn.BatchNorm2d(planes//2)
        self.quantize_activation_1_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//2)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv_1 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_1 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_2 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_2 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_3 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_3 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_4 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_4 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_4 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out1 = self.conv1_1(x[:, :self.half_max_ic*2])
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)
        out1 = self.quantize_activation_1_1(out1)
        out2 = self.conv1_2(x[:, self.half_max_ic*2:])
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)
        out2 = self.quantize_activation_1_2(out2)
        out = torch.cat((out1, out2), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.fa_res:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        identity1 = identity[:, :self.half_max_ic]
        identity2 = identity[:, self.half_max_ic:self.half_max_ic*2]
        identity3 = identity[:, self.half_max_ic*2:self.half_max_ic*3]
        identity4 = identity[:, self.half_max_ic*3:]

        out1 = out[:, :self.half_max_ic]
        out2 = out[:, self.half_max_ic:self.half_max_ic*2]
        out3 = out[:, self.half_max_ic*2:self.half_max_ic*3]
        out4 = out[:, self.half_max_ic*3:]

        out1 = self.concat_conv_1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn_1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv_2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn_2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out3 = self.concat_conv_3(torch.cat((out3, identity3), dim=1))
        out3 = self.concat_bn_3(out3)
        out3 = self.relu(out3)
        out3 = self.concat_quantize_activation_3(out3)
        out4 = self.concat_conv_4(torch.cat((out4, identity4), dim=1))
        out4 = self.concat_bn_4(out4)
        out4 = self.relu(out4)
        out4 = self.concat_quantize_activation_4(out4)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.fa_res:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class QuantizeBottleneckV5(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV5, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_max_ic = 256
        self.conv1_1 = DorefaConv2d(inplanes//2, planes//2, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_1 = nn.BatchNorm2d(planes//2)
        self.quantize_activation_1_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//2)
        self.conv1_2 = DorefaConv2d(inplanes//2, planes//2, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_2 = nn.BatchNorm2d(planes//2)
        self.quantize_activation_1_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//2)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv_1 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_1 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_2 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_2 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_3 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_3 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_4 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_4 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_4 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_5 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_5 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_5 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_6 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_6 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_6 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_7 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_7 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_7 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_8 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_8 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_8 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out1 = self.conv1_1(x[:, :self.half_max_ic*2])
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)
        out1 = self.quantize_activation_1_1(out1)
        out2 = self.conv1_2(x[:, self.half_max_ic*2:])
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)
        out2 = self.quantize_activation_1_2(out2)
        out = torch.cat((out1, out2), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.fa_res:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        identity1 = identity[:, :self.half_max_ic]
        identity2 = identity[:, self.half_max_ic:self.half_max_ic*2]
        identity3 = identity[:, self.half_max_ic*2:self.half_max_ic*3]
        identity4 = identity[:, self.half_max_ic*3:self.half_max_ic*4]
        identity5 = identity[:, self.half_max_ic*4:self.half_max_ic*5]
        identity6 = identity[:, self.half_max_ic*5:self.half_max_ic*6]
        identity7 = identity[:, self.half_max_ic*6:self.half_max_ic*7]
        identity8 = identity[:, self.half_max_ic*7:]

        out1 = out[:, :self.half_max_ic]
        out2 = out[:, self.half_max_ic:self.half_max_ic*2]
        out3 = out[:, self.half_max_ic*2:self.half_max_ic*3]
        out4 = out[:, self.half_max_ic*3:self.half_max_ic*4]
        out5 = out[:, self.half_max_ic*4:self.half_max_ic*5]
        out6 = out[:, self.half_max_ic*5:self.half_max_ic*6]
        out7 = out[:, self.half_max_ic*6:self.half_max_ic*7]
        out8 = out[:, self.half_max_ic*7:]

        out1 = self.concat_conv_1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn_1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv_2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn_2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out3 = self.concat_conv_3(torch.cat((out3, identity3), dim=1))
        out3 = self.concat_bn_3(out3)
        out3 = self.relu(out3)
        out3 = self.concat_quantize_activation_3(out3)
        out4 = self.concat_conv_4(torch.cat((out4, identity4), dim=1))
        out4 = self.concat_bn_4(out4)
        out4 = self.relu(out4)
        out4 = self.concat_quantize_activation_4(out4)
        out5 = self.concat_conv_5(torch.cat((out5, identity5), dim=1))
        out5 = self.concat_bn_5(out5)
        out5 = self.relu(out5)
        out5 = self.concat_quantize_activation_5(out5)
        out6 = self.concat_conv_6(torch.cat((out6, identity6), dim=1))
        out6 = self.concat_bn_6(out6)
        out6 = self.relu(out6)
        out6 = self.concat_quantize_activation_6(out6)
        out7 = self.concat_conv_7(torch.cat((out7, identity7), dim=1))
        out7 = self.concat_bn_7(out7)
        out7 = self.relu(out7)
        out7 = self.concat_quantize_activation_7(out7)
        out8 = self.concat_conv_8(torch.cat((out8, identity8), dim=1))
        out8 = self.concat_bn_8(out8)
        out8 = self.relu(out8)
        out8 = self.concat_quantize_activation_8(out8)
        out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), dim=1)

        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.fa_res:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class QuantizeBottleneckV6(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, fa_res=True, is_train=True):
        super(QuantizeBottleneckV6, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_max_ic = 256
        self.conv1_1 = DorefaConv2d(inplanes//4, planes//4, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_1 = nn.BatchNorm2d(planes//4)
        self.quantize_activation_1_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//4)
        self.conv1_2 = DorefaConv2d(inplanes//4, planes//4, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_2 = nn.BatchNorm2d(planes//4)
        self.quantize_activation_1_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//4)
        self.conv1_3 = DorefaConv2d(inplanes//4, planes//4, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_3 = nn.BatchNorm2d(planes//4)
        self.quantize_activation_1_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//4)
        self.conv1_4 = DorefaConv2d(inplanes//4, planes//4, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1_4 = nn.BatchNorm2d(planes//4)
        self.quantize_activation_1_4 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes//4)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv3 = DorefaConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes*self.expansion)
        self.concat_conv_1 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_1 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_2 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_2 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_3 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_3 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_4 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_4 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_4 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_5 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_5 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_5 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_6 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_6 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_6 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_7 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_7 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_7 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.concat_conv_8 = DorefaConv2d(self.half_max_ic*2, self.half_max_ic, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn_8 = nn.BatchNorm2d(self.half_max_ic)
        self.concat_quantize_activation_8 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_max_ic)
        self.downsample = downsample
        self.stride = stride
        self.fa_res = fa_res
        self.is_train = is_train

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes*self.expansion)
        self.feature_adaptation_4_1 = conv1x1(planes*self.expansion, planes)
        self.feature_adaptation_4_2 = conv3x3(planes, planes)
        self.feature_adaptation_4_3 = conv1x1(planes, planes*self.expansion)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.fc4_1 = nn.Linear(planes*self.expansion, planes//squeeze_factor)
        self.fc4_2 = nn.Linear(planes//squeeze_factor, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            teacher_middle_outputs = x[0]
            x = x[1]
        identity = x

        out1 = self.conv1_1(x[:, :self.half_max_ic*2])
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)
        out1 = self.quantize_activation_1_1(out1)
        out2 = self.conv1_2(x[:, self.half_max_ic*2:self.half_max_ic*4])
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)
        out2 = self.quantize_activation_1_2(out2)
        out3 = self.conv1_3(x[:, self.half_max_ic*4:self.half_max_ic*6])
        out3 = self.bn1_3(out3)
        out3 = self.relu(out3)
        out3 = self.quantize_activation_1_3(out3)
        out4 = self.conv1_4(x[:, self.half_max_ic*6:])
        out4 = self.bn1_4(out4)
        out4 = self.relu(out4)
        out4 = self.quantize_activation_1_4(out4)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[0])
            attention_out = self.fc1_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc1_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_1 = self.feature_adaptation_1_1(attention_out)
            feature_adaptation_1 = self.feature_adaptation_1_2(feature_adaptation_1)
            feature_adaptation_1 = self.feature_adaptation_1_3(feature_adaptation_1)
            if self.fa_res:
                feature_adaptation_1 = feature_adaptation_1 + out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quantize_activation_2(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[1])
            attention_out = self.fc2_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc2_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_2 = self.feature_adaptation_2_1(attention_out)
            feature_adaptation_2 = self.feature_adaptation_2_2(feature_adaptation_2)
            feature_adaptation_2 = self.feature_adaptation_2_3(feature_adaptation_2)
            if self.fa_res:
                feature_adaptation_2 = feature_adaptation_2 + out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.quantize_activation_3(out)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[2])
            attention_out = self.fc3_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc3_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_3 = self.feature_adaptation_3_1(attention_out)
            feature_adaptation_3 = self.feature_adaptation_3_2(feature_adaptation_3)
            feature_adaptation_3 = self.feature_adaptation_3_3(feature_adaptation_3)
            if self.fa_res:
                feature_adaptation_3 = feature_adaptation_3 + out

        if self.downsample is not None:
            identity = self.downsample(x)

        identity1 = identity[:, :self.half_max_ic]
        identity2 = identity[:, self.half_max_ic:self.half_max_ic*2]
        identity3 = identity[:, self.half_max_ic*2:self.half_max_ic*3]
        identity4 = identity[:, self.half_max_ic*3:self.half_max_ic*4]
        identity5 = identity[:, self.half_max_ic*4:self.half_max_ic*5]
        identity6 = identity[:, self.half_max_ic*5:self.half_max_ic*6]
        identity7 = identity[:, self.half_max_ic*6:self.half_max_ic*7]
        identity8 = identity[:, self.half_max_ic*7:]

        out1 = out[:, :self.half_max_ic]
        out2 = out[:, self.half_max_ic:self.half_max_ic*2]
        out3 = out[:, self.half_max_ic*2:self.half_max_ic*3]
        out4 = out[:, self.half_max_ic*3:self.half_max_ic*4]
        out5 = out[:, self.half_max_ic*4:self.half_max_ic*5]
        out6 = out[:, self.half_max_ic*5:self.half_max_ic*6]
        out7 = out[:, self.half_max_ic*6:self.half_max_ic*7]
        out8 = out[:, self.half_max_ic*7:]

        out1 = self.concat_conv_1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn_1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv_2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn_2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out3 = self.concat_conv_3(torch.cat((out3, identity3), dim=1))
        out3 = self.concat_bn_3(out3)
        out3 = self.relu(out3)
        out3 = self.concat_quantize_activation_3(out3)
        out4 = self.concat_conv_4(torch.cat((out4, identity4), dim=1))
        out4 = self.concat_bn_4(out4)
        out4 = self.relu(out4)
        out4 = self.concat_quantize_activation_4(out4)
        out5 = self.concat_conv_5(torch.cat((out5, identity5), dim=1))
        out5 = self.concat_bn_5(out5)
        out5 = self.relu(out5)
        out5 = self.concat_quantize_activation_5(out5)
        out6 = self.concat_conv_6(torch.cat((out6, identity6), dim=1))
        out6 = self.concat_bn_6(out6)
        out6 = self.relu(out6)
        out6 = self.concat_quantize_activation_6(out6)
        out7 = self.concat_conv_7(torch.cat((out7, identity7), dim=1))
        out7 = self.concat_bn_7(out7)
        out7 = self.relu(out7)
        out7 = self.concat_quantize_activation_7(out7)
        out8 = self.concat_conv_8(torch.cat((out8, identity8), dim=1))
        out8 = self.concat_bn_8(out8)
        out8 = self.relu(out8)
        out8 = self.concat_quantize_activation_8(out8)
        out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), dim=1)
        if self.is_train:
            attention_out = self.downsample_and_squeeze(teacher_middle_outputs[3])
            attention_out = self.fc4_1(attention_out)
            attention_out = self.relu(attention_out)
            attention_out = self.fc4_2(attention_out)
            attention_out = self.sigmoid(attention_out)
            attention_out = attention_out.unsqueeze(2).unsqueeze(3)
            attention_out = attention_out.expand_as(out) * out
            feature_adaptation_4 = self.feature_adaptation_4_1(attention_out)
            feature_adaptation_4 = self.feature_adaptation_4_2(feature_adaptation_4)
            feature_adaptation_4 = self.feature_adaptation_4_3(feature_adaptation_4)
            if self.is_train:
                feature_adaptation_4 = feature_adaptation_4 + out

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, feature_adaptation_4, out]

        return out


class ArchNetResNet(nn.Module):
    def __init__(self, block_normal, block_split_for_bottleneck2, block_split_for_bottleneck3_1, block_split_for_bottleneck3_2, block_split_for_bottleneck4_1, block_split_for_bottleneck4_2, layers, weight_dtype, feature_dtype, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, squeeze_factor=8, distillation_idx=None, fa_res=True, is_train=True):
        super(ArchNetResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.w_dtype = weight_dtype
        self.f_dtype = feature_dtype
        self.conv1 = DorefaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(64)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=64)
        self.conv2 = DorefaConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(64)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=64)
        self.conv3 = DorefaConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(64)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block_normal, 64, layers[0], squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train)
        self.layer2 = self._make_layer(block_split_for_bottleneck2, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train)
        self.layer3 = self._make_layer_2_blocks(block_split_for_bottleneck3_1, block_split_for_bottleneck3_2, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], squeeze_factor=squeeze_factor, is_last_stage=False, fa_res=fa_res, is_train=is_train)
        self.layer4 = self._make_layer_2_blocks(block_split_for_bottleneck4_1, block_split_for_bottleneck4_2, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], squeeze_factor=squeeze_factor, is_last_stage=True, fa_res=fa_res, is_train=is_train)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DorefaLinear1d(2048, 1000, weight_dtype=DATA_TYPE_DICT['uint8'], bias=True, quant_weight_mode='gaussian')
        self.quantize_activation_fc = DorefaQuantizeActivation(feature_dtype=DATA_TYPE_DICT['int16'], out_channels=1000, out_scale=32, after_bn=False)
        self.distillation_idx = distillation_idx
        self.is_train = is_train

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, squeeze_factor=8, fa_res=True, is_train=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                DorefaConv2d(self.inplanes, planes*block.expansion, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=self.w_dtype),
                nn.BatchNorm2d(planes*block.expansion),
                nn.ReLU(inplace=True),
                DorefaQuantizeActivation(feature_dtype=self.f_dtype, out_channels=64))

        layers = []
        layers.append(block(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))

        return nn.Sequential(*layers)

    def _make_layer_2_blocks(self, block1, block2, planes, blocks, stride=1, dilate=False, squeeze_factor=8, is_last_stage=False, fa_res=True, is_train=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block1.expansion:
            if not is_last_stage:
                downsample = nn.Sequential(
                    DorefaConv2d(self.inplanes, planes*block1.expansion, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=self.w_dtype),
                    nn.BatchNorm2d(planes*block1.expansion),
                    nn.ReLU(inplace=True),
                    DorefaQuantizeActivation(feature_dtype=self.f_dtype, out_channels=64))
            else:
                downsample = SplitDownsample(block1, planes, stride, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype)

        layers = []
        layers.append(block1(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))
        self.inplanes = planes * block1.expansion
        for _ in range(1, blocks):
            layers.append(block2(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if self.is_train:
            all_middle_outputs = []
            if isinstance(x, list):
                teacher_middle_outputs = x[0]
                x = x[1]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.quantize_activation_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.quantize_activation_2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.quantize_activation_3(x)
        x = self.maxpool(x)

        if self.is_train:
            all_middle_outputs.append(x)

        block_count = 0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block_count += 1
                if self.is_train:
                    tmp_teacher_middle_outputs = teacher_middle_outputs[4*block_count-3:4*block_count+1]
                    outs = block([tmp_teacher_middle_outputs, x])
                    x = outs[-1]
                    all_middle_outputs.extend(outs[:-1])
                else:
                    x = block(x)

        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        x = self.quantize_activation_fc(x)
        if self.is_train:
            all_middle_outputs.append(x)
            return all_middle_outputs[:self.distillation_idx+1]
        else:
            return x

    def forward(self, x):
        return self._forward_impl(x)


def archnet_resnet50(weight_bit, feature_bit, squeeze_factor=24, distillation_idx=None, fa_res=True, is_train=True, **kwargs):
    bit_map_dict = {2: 'uint2', 4: 'uint4', 8: 'uint8'}
    if weight_bit not in bit_map_dict or feature_bit not in bit_map_dict:
        assert False, 'Not supported bit width'

    model = ArchNetResNet(block_normal=QuantizeBottleneckV1, block_split_for_bottleneck2=QuantizeBottleneckV2, block_split_for_bottleneck3_1=QuantizeBottleneckV3, block_split_for_bottleneck3_2=QuantizeBottleneckV4, block_split_for_bottleneck4_1=QuantizeBottleneckV5, block_split_for_bottleneck4_2=QuantizeBottleneckV6, layers=[3, 4, 6, 3], weight_dtype=DATA_TYPE_DICT[bit_map_dict[weight_bit]], feature_dtype=DATA_TYPE_DICT[bit_map_dict[feature_bit]], squeeze_factor=squeeze_factor, distillation_idx=distillation_idx, fa_res=fa_res, is_train=is_train, **kwargs)

    return model
