#!/usr/bin/env python
# coding=utf-8
"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum

__all__ = ['Qmodes', 'log_shift', '_Conv2dQ', '_LinearQ', '_ActQ',
           'update_running_scale', 'ln_error', 'truncation', 'round_cus',
           'get_sparsity_mask', 'FunStopGradient', 'round_pass', 'grad_scale']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def get_sparsity_mask(param, sparsity):
    bottomk, _ = torch.topk(param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True)
    threshold = bottomk.data[-1]  # This is the largest elemet from the group of elements that we prune away
    return torch.gt(torch.abs(param), threshold).type(param.type())


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class FunStopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        stopGradientMask, = ctx.saved_tensors
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def round_cus(t, inplace=False):
    tt = t.clone()
    tt[tt > 0] = 1.0
    tt[tt < 0] = -1.0 + 2 ** -16
    temp = (t + 0.5 * tt).type(torch.int32).type_as(t)
    if inplace:
        t.copy_(temp)
        return t
    return temp


def update_running_scale(data_fp, scale_old, error, Qn, Qp, qmode=Qmodes.layer_wise, is_l2=True):
    s_error = ln_error(data_fp, scale_old / 2, Qn, Qp, qmode, is_l2)
    b_error = ln_error(data_fp, scale_old * 2, Qn, Qp, qmode, is_l2)
    a1 = error - s_error
    a2 = b_error - error
    g1 = a1 >= 0
    g2 = a2 > 0
    g3 = a1 + a2 >= 0
    """
                    g1  g2  g3  res
                    0   0   0   big
                    0   0   1   big
                    0   1   0   keep
                    0   1   1   keep
                    1   0   0   big
                    1   0   1   small
                    1   1   0   small
                    1   1   1   small
    """
    b = ((g1 == 0) * (g2 == 0) == 1) + ((g1 * (g2 == 0) * (g3 == 0)) > 0) > 0
    s = (((g1 * g2) > 0) + ((g1 * (g2 == 0) * g3) > 0)) > 0
    return b, s


def ln_error(x, scale, Qn, Qp, qmode=Qmodes.layer_wise, is_l2=True):
    x_clip = (x / scale).clamp(Qn, Qp)
    x_q = x_clip.round()
    x_q = x_q * scale
    if qmode == Qmodes.layer_wise:
        if is_l2:
            error = ((x - x_q) ** 2).sum() / x.reshape(-1).size()[0]
        else:
            error = (x - x_q).abs().sum() / x.reshape(-1).size()[0]
    else:
        if is_l2:
            error = ((x - x_q) ** 2).sum(dim=0) / x.shape[0]
        else:
            error = (x - x_q).abs().sum(dim=0) / x.shape[0]
    # x_clip = x_clip * scale
    return error


def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        default.update({
            'signed': True})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.signed = kwargs_q['signed']
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)
