#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F
import math
from .lsq_modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ


__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        return F.linear(x, w_q, self.bias)


class ActLSQ(_ActQ):
    def __init__(self, nbits=4, signed=False):
        super(ActLSQ, self).__init__(nbits=nbits, signed=signed)

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.signed:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha

        return x


class EmbeddingLSQ(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, weight_bit):
        super(EmbeddingLSQ, self).__init__()
        self.padding_idx = padding_idx
        self.linear = LinearLSQ(num_embeddings, embedding_dim, bias=False, nbits=weight_bit)

    def forward(self, x):
        onehot_to_idx = torch.topk(x, 1)[1]
        mask = onehot_to_idx.eq(self.padding_idx)

        x = self.linear(x)
        x = x.masked_fill(mask, value=0)

        return x
