import numpy as np
import torch
import torch.nn as nn

def prox_positive(x):
    return torch.nn.functional.threshold(x,0,0)


def prox_identity(x):
    return x


def prox_soft_threshold(x,lamb):
    return torch.sign(x)*torch.nn.functional.threshold(torch.abs(x)-lamb,0,0)


def prox_hard_threshold(x,k):
    # hard-threshold each row of x
    x = x.clone().detach().cpu()
    m = x.data.shape[1]
    a,_ = torch.abs(x).data.sort(dim=1,descending=True)
    thresh = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))))
    mask = torch.tensor((np.abs(x.data.cpu().numpy())>thresh.cpu().numpy()) + 0.,dtype=torch.float)
    return (x*mask).to(device)


def prox_normalization(x):
    norm = torch.norm(x, p=2).detach().item()
#     if norm >= 1:
#         return x/norm
    return x/norm


def L1Norm(params: list):
    norm = 0
    for param in params:
        norm += np.abs(param.data.cpu().numpy()).sum()
    return norm


def L2Norm(params: list):
    norm_sq = 0
    for param in params:
        norm_sq += np.linalg.norm(param.data.cpu().numpy())**2
    return norm_sq


def conv_sparse_coder(im: torch.Tensor, atoms: list, comp_idx: list):
    x = 0
    for indx in comp_idx:
        x += atoms[indx]
    return x


def get_atoms(convs: list, maps: list):
    atoms = []
    n_components = len(convs)
    for indx in range(n_components):
        atoms.append(convs[indx](maps[indx]))
    return atoms


def get_residual(im: torch.Tensor, atoms: list):
    recon = 0
    for atom in atoms:
        recon += atom
    return im - recon    
