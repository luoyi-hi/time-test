import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, st=None):
        if st == None:
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            else: raise NotImplementedError
            return x
        else:
            if mode == 'norm':
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            else: raise NotImplementedError
            return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def _normalize2(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            a = self.affine_weight.detach()
            b = self.affine_bias.detach()
            x = x * a
            x = x + b
        return x

    def _denormalize2(self, x):
        if self.affine:
            a = self.affine_weight.detach()
            b = self.affine_bias.detach()
            x = x - b
            x = x / (a + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
