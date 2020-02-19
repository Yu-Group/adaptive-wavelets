import numpy as np
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from funcs import *

class Conv_SpCoding(nn.Module):
    def __init__(self, kernel_size, n_dim, n_comp, stride=1):
        super(Conv_SpCoding, self).__init__()
        self.kernel_size = kernel_size
        self.n_dim = n_dim
        self.n_comp = n_comp
        
        # initialize filters
        torch.manual_seed(10)
        self.convs = nn.ModuleList([nn.Conv2d(1,1,kernel_size=kernel_size,stride=stride,bias=False) for i in range(n_comp)])
        # normalization
        for conv in self.convs:
            conv.weight.data = prox_normalization(conv.weight.data)        

    def forward(self):
        x = 0
        for i in range(len(self.convs)):
            x += self.convs[i](self.maps[i])
        return x
    
    def init_maps(self, n_batch):
        self.maps = nn.ParameterList([nn.Parameter(torch.zeros(n_batch,1,self.n_dim,self.n_dim)) for i in range(self.n_comp)])


class NMF(nn.Module):
    def __init__(self, n_obs, n_dim, n_comp):
        super(NMF, self).__init__()
        torch.manual_seed(10)
        self.D = nn.Parameter(torch.randn(n_comp, n_dim))
        torch.manual_seed(10)
        self.W = nn.Parameter(torch.randn(n_obs, n_comp))

        # positivity
        self.D.data = prox_positive(self.D.data)
        self.W.data = prox_positive(self.W.data)

    def forward(self, indices=None):
        if indices is None:
            X = torch.matmul(self.W, self.D)
        else:
            X = torch.matmul(self.W[indices], self.D)
        return X


class Blockwise_SGD(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, **kwargs)

    def step(self, indx_block, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if indx_block >= len(self.param_groups):
            raise ValueError("Block index exceeds the total number of blocks")

        group = self.param_groups[indx_block]

        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'], d_p)
        return loss


class Optimizer(Blockwise_SGD):
    def __init__(self, params, proxs, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError("Invalid length of argument proxs: {} instead of {}".format(len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, indx_block, closure=None):
        # perform a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(indx_block, closure=closure)

        group = self.param_groups[indx_block]
        prox = group['prox']

        # apply the proximal operator to each parameter in a group
        for p in group['params']:
            p.data = prox(p.data)

            
def csc_optimizer(csc, lr_c, lr_w, lamb):
    param_list = [{'params': csc.convs.parameters(), 'lr': lr_c},
                  {'params': csc.maps.parameters(), 'lr': lr_w}]
    prox_list = [prox_normalization, partial(prox_soft_threshold, lamb=lr_w*lamb)]
    optimizer = Optimizer(param_list, prox_list, momentum=0.0)   
    return optimizer