import numpy as np
import torch
import torch.nn as nn


class Attributer(nn.Module):
    '''Get attribution scores for wavelet coefficients
    Params
    ------
    mt: nn.Module
        model after all the transformations
    attr_methods: str
        currently support InputXGradient only
    device: str
        use GPU or CPU
    '''

    def __init__(self, mt, attr_methods='InputXGradient', is_train=True, device='cuda'):
        super().__init__()
        self.mt = mt.to(device)
        self.attr_methods = attr_methods
        self.is_train = is_train
        self.device = device

    def forward(self, x: tuple, target=1, additional_forward_args=None):
        if self.attr_methods == 'InputXGradient':
            attributions = self.InputXGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'IntegratedGradient':
            attributions = self.IntegratedGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'Saliency':
            attributions = self.Saliency(x, target, additional_forward_args)
        else:
            raise ValueError
        return attributions

    def InputXGradient(self, x: tuple, target=1, additional_forward_args=None):
        if target != -1:
            outputs = self.mt(x, additional_forward_args)[:, target]
            if self.is_train:
                grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)
            else:
                grads = torch.autograd.grad(torch.unbind(outputs), x)
                # input * gradient
            attributions = tuple(xi * gi for xi, gi in zip(x, grads))
        else:
            attributions = ()
            for target in range(10):
                outputs = self.mt(x, additional_forward_args)[:, target]
                if self.is_train:
                    grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)
                else:
                    grads = torch.autograd.grad(torch.unbind(outputs), x)
                    # input * gradient
                attributions += tuple(xi * gi for xi, gi in zip(x, grads))
        return attributions

    def Saliency(self, x: tuple, target=1, additional_forward_args=None):
        if target != -1:
            outputs = self.mt(x, additional_forward_args)[:, target]
            if self.is_train:
                grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)
            else:
                grads = torch.autograd.grad(torch.unbind(outputs), x)
            attributions = grads
        else:
            attributions = ()
            for target in range(10):
                outputs = self.mt(x, additional_forward_args)[:, target]
                if self.is_train:
                    grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)
                else:
                    grads = torch.autograd.grad(torch.unbind(outputs), x)
                attributions += grads
        return attributions

    ### TO DO!! ###
    # implement batch version of IG
    def IntegratedGradient(self, x: tuple, target=1, additional_forward_args=None, M=100):
        n = len(x)
        mult_grid = np.array(range(M)) / (M - 1)  # fractions to multiply by

        # compute all the input vecs
        input_vecs = []
        baselines = []
        for i in range(n):
            baselines.append(torch.zeros_like(x[i]))  # baseline of zeros
            shape = list(x[i].shape[1:])
            shape.insert(0, M)
            inp = torch.empty(shape, dtype=torch.float32, requires_grad=True).to(self.device)
            for j, prop in enumerate(mult_grid):
                inp[j] = baselines[i] + prop * (x[i] - baselines[i])
            inp.retain_grad()
            input_vecs.append(inp)

        # run forward pass
        output = self.mt(input_vecs, additional_forward_args)[:, 1].sum()
        output.backward(retain_graph=True)

        # ig
        scores = []
        for i in range(n):
            imps = input_vecs[i].grad.mean(0) * (x[i] - baselines[i])  # record all the grads
            scores.append(imps)
        return tuple(scores)
