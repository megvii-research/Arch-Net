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


class QuantizeBasicBlockV1(nn.Module):
    # according to the original paper, the limitation of the number of channel is 512.
    # This block is for the blocks that do not need to split the channel
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, zero_weight_mode=False, fa_res=True, is_train=True):
        super(QuantizeBasicBlockV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = DorefaConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.concat_conv = DorefaConv2d(planes*2, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn3 = nn.BatchNorm2d(planes)
        self.quantize_activation_3 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.downsample = downsample
        self.stride = stride

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fa_res = fa_res
        self.is_train = is_train

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.cat((out, identity), dim=1)
        out = self.concat_conv(out)
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

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, out]

        return out


class QuantizeBasicBlockV2(nn.Module):
    # according to the original paper, the limitation of the number of channel is 512.
    # This block is for the blocks that do not need to split the channel
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, weight_dtype=None, feature_dtype=None, squeeze_factor=8, zero_weight_mode=False, fa_res=True, is_train=True):
        super(QuantizeBasicBlockV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.half_inplanes = inplanes // 2
        self.half_planes = planes // 2
        self.conv1 = DorefaConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.conv2 = DorefaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=planes)
        self.concat_conv1 = DorefaConv2d(planes, self.half_planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn1 = nn.BatchNorm2d(self.half_planes)
        self.concat_quantize_activation_1 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_planes)
        self.concat_conv2 = DorefaConv2d(planes, self.half_planes, kernel_size=3, stride=1, padding=1, bias=False, weight_dtype=weight_dtype)
        self.concat_bn2 = nn.BatchNorm2d(self.half_planes)
        self.concat_quantize_activation_2 = DorefaQuantizeActivation(feature_dtype=feature_dtype, out_channels=self.half_planes)
        self.downsample = downsample
        self.stride = stride

        self.feature_adaptation_1_1 = conv1x1(planes, planes)
        self.feature_adaptation_1_2 = conv3x3(planes, planes)
        self.feature_adaptation_1_3 = conv1x1(planes, planes)
        self.feature_adaptation_2_1 = conv1x1(planes, planes)
        self.feature_adaptation_2_2 = conv3x3(planes, planes)
        self.feature_adaptation_2_3 = conv1x1(planes, planes)
        self.feature_adaptation_3_1 = conv1x1(planes, planes)
        self.feature_adaptation_3_2 = conv3x3(planes, planes)
        self.feature_adaptation_3_3 = conv1x1(planes, planes)
        self.fc1_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc1_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc2_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc2_2 = nn.Linear(planes//squeeze_factor, planes)
        self.fc3_1 = nn.Linear(planes, planes//squeeze_factor)
        self.fc3_2 = nn.Linear(planes//squeeze_factor, planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fa_res = fa_res
        self.is_train = is_train

    def downsample_and_squeeze(self, input_tensor):
        out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
        out = torch.nn.Flatten()(out)

        return out

    def forward(self, x):
        if self.is_train:
            if isinstance(x, list):
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

        if self.downsample is not None:
            identity = self.downsample(identity)
        identity1 = identity[:, :self.half_planes]
        identity2 = identity[:, self.half_planes:]

        out1 = out[:, :self.half_planes]
        out2 = out[:, self.half_planes:]

        out1 = self.concat_conv1(torch.cat((out1, identity1), dim=1))
        out1 = self.concat_bn1(out1)
        out1 = self.relu(out1)
        out1 = self.concat_quantize_activation_1(out1)
        out2 = self.concat_conv2(torch.cat((out2, identity2), dim=1))
        out2 = self.concat_bn2(out2)
        out2 = self.relu(out2)
        out2 = self.concat_quantize_activation_2(out2)
        out = torch.cat((out1, out2), dim=1)
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

            return [feature_adaptation_1, feature_adaptation_2, feature_adaptation_3, out]

        return out


class ArchNetResNet(nn.Module):
    def __init__(self, block_normal, block_split, layers, weight_dtype, feature_dtype, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, squeeze_factor=8, distillation_idx=None, fa_res=True, is_train=True):
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
        self.layer2 = self._make_layer(block_normal, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train)
        self.layer3 = self._make_layer(block_normal, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train)
        self.layer4 = self._make_layer(block_split, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DorefaLinear1d(512, 1000, weight_dtype=DATA_TYPE_DICT['uint8'], bias=True, quant_weight_mode='gaussian')
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
        layers.append(block(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, weight_dtype=self.w_dtype, feature_dtype=self.f_dtype, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, squeeze_factor=squeeze_factor, fa_res=fa_res, is_train=is_train))

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
                    tmp_teacher_middle_outputs = teacher_middle_outputs[3*block_count-2:3*block_count+1]
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


def archnet_resnet18(weight_bit, feature_bit, squeeze_factor=24, distillation_idx=None, fa_res=True, is_train=True, **kwargs):
    bit_map_dict = {2: 'uint2', 4: 'uint4', 8: 'uint8'}
    if weight_bit not in bit_map_dict or feature_bit not in bit_map_dict:
        assert False, 'Not supported bit width'

    model = ArchNetResNet(QuantizeBasicBlockV1, QuantizeBasicBlockV2, [2, 2, 2, 2], weight_dtype=DATA_TYPE_DICT[bit_map_dict[weight_bit]], feature_dtype=DATA_TYPE_DICT[bit_map_dict[feature_bit]], squeeze_factor=squeeze_factor, distillation_idx=distillation_idx, fa_res=fa_res, is_train=is_train, **kwargs)

    return model
