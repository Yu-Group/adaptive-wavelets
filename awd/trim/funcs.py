import sys

import numpy as np
import torch

sys.path.append('..')


def prox_positive(x):
    return torch.nn.functional.threshold(x, 0, 0)


def prox_identity(x):
    return x


def prox_soft_threshold(x, lamb):
    return torch.sign(x) * torch.nn.functional.threshold(torch.abs(x) - lamb, 0, 0)


def prox_hard_threshold(x, k):
    # hard-threshold each row of x
    x = x.clone().detach().cpu()
    m = x.data.shape[1]
    a, _ = torch.abs(x).data.sort(dim=1, descending=True)
    thresh = torch.mm(a[:, k].unsqueeze(1), torch.Tensor(np.ones((1, m))))
    mask = torch.tensor((np.abs(x.data.cpu().numpy()) > thresh.cpu().numpy()) + 0., dtype=torch.float)
    return (x * mask).to(device)


def prox_normalization(x):
    '''
    x : (B,C,H,W) tensor
    '''
    norm = torch.norm(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
    return x / norm
