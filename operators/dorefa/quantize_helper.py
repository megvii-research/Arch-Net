#!/usr/bin/env python
# coding=utf-8
import torch
from .data_type import dorefa_uint16, dorefa_int16


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        offset = 0 if dtype in [dorefa_int16, dorefa_uint16] else 0.5
        y = (x + offset) // 1
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class QuantWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        scale = x.abs().mean() * 2
        multiplier = dtype.max_val
        upper_bound = 0.5
        lower_bound = upper_bound - 1

        x = torch.clamp(x / scale, lower_bound, upper_bound) - lower_bound  # [0, 1]
        y = ((x * multiplier + 0.5) // 1) / multiplier + lower_bound  # [lower, upper]
        return y * scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def quant_weight(x, dtype, mode):
    if mode.lower() == "gaussian":
        x = x.tanh()
        if len(x.shape) == 2:
            x = x - torch.mean(x, dim=1, keepdim=True)
            scale = x.view(x.shape[0], -1).abs().max(dim=1)[0].view(x.shape[0], 1) * 2
        elif len(x.shape) == 4:
            x = x - torch.mean(x, dim=[1, 2, 3], keepdim=True)
            scale = (
                x.view(x.shape[0], -1).abs().max(dim=1)[0].view(x.shape[0], 1, 1, 1) * 2
            )
        x = (x / scale + 0.5) * dtype.max_val
        x_qt = (RoundSTE.apply(x, dtype) / dtype.max_val - 0.5) * scale
    elif mode.lower() == "normal":
        x = x.tanh()
        x_qt = QuantWeight.apply(x, dtype)
        scale = None
    else:
        assert (
            False
        ), "only support quant_weight_mode in ['normal', 'gaussian']"
    return x_qt, scale


def quant_feature(x, dtype, scale):
    if len(torch.unique(scale)) > 1 or (
        len(torch.unique(scale)) == 1 and torch.unique(scale).item() != 1.0
    ):
        assert dtype in [
            dorefa_int16,
            dorefa_uint16,
        ], "out_scale != 1.0 and multi-out_scale only support feature_dtype in ['dorefa_int16', 'dorefa_uint16']"
    scale = scale.to(x.device)
    if x.dim() == 2:
        scale = scale.view(1, -1)
    if x.dim() == 4:
        scale = scale.view(1, -1, 1, 1)
    is_quant = False if dtype in [dorefa_int16, dorefa_uint16] else True
    offset = 0.5 if dtype.is_signed is True else 0
    if is_quant:
        x = torch.clamp(x + offset, 0, 1)
        x = x * dtype.max_val
        x = RoundSTE.apply(x, dtype) - offset * dtype.max_val
        x = x / dtype.max_val
    else:
        offset *= scale
        x = torch.clamp_min(torch.min(x + offset, scale), 0) - offset
    return x
