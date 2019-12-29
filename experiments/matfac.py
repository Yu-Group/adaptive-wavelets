import numpy as np
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from funcs import *

class Filter(nn.Module):
    def __init__(self, kernel_size, n_comp):
        super(Filter, self).__init__()
        torch.manual_seed(10)
        self.convs = nn.ModuleList([nn.Conv2d(1,1,kernel_size,bias=False) for i in range(n_comp)])

        # normalization
        for conv in self.convs:
            conv.weight.data = prox_normalization(conv.weight.data)

    def forward(self, maps: list):
        x = 0
        for i in range(len(maps)):
            x += self.convs[i](maps[i])
        return x


class FeatureMap(nn.Module):
    def __init__(self, n_dim, n_comp):
        super(FeatureMap, self).__init__()
        torch.manual_seed(10)
        self.maps = nn.ParameterList([nn.Parameter(torch.randn(1,1,n_dim,n_dim)) for i in range(n_comp)])

    def forward(self, convs: list):
        x = 0
        for i in range(len(convs)):
            x += convs[i](self.maps[i])
        return x


class Optimizer(SGD):
    def __init__(self, params, proxs, lr=required, momentum=0, dampening=0, nesterov=False):

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0, nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError("Invalid length of argument proxs: {} instead of {}".format(len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, closure=None):
        # perform a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)

        for group in self.param_groups:
            prox = group['prox']

            # apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(p.data)
