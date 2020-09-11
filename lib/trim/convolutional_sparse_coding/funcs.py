import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from copy import deepcopy

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


def unfreeze(module, param='dict', obj_type='csc'):
    if obj_type == 'csc':
        if param == 'dict':
            module.convs.requires_grad_(True)
            module.maps.requires_grad_(False)
        elif param == 'map':
            module.convs.requires_grad_(False)
            module.maps.requires_grad_(True)
    elif obj_type == 'nmf':
        if param == 'dict':
            module.D.requires_grad_(True)
            module.W.requires_grad_(False)
        elif param == 'map':
            module.D.requires_grad_(False)
            module.W.requires_grad_(True)
    else:
        print('invalid arguments')


def L1Reg_loss(module, X, lamb, lamb_cd=0, model=None, comp_indx=None):
    X_ = module()
    n_batch = X.shape[0]
    reg_loss = (torch.norm(X-X_)**2/(2*n_batch)).data.item()
    reg_loss += lamb*L1Norm(module.maps.parameters())
    if lamb_cd > 0:
        atoms = get_atoms(module)
        reg_loss += lamb_cd * cd.cd(X, model=model, mask=None, model_type='resnet18', device='cuda',
                            transform=partial(conv_transform, atoms=atoms, comp_indx=comp_indx))[0].flatten()[1].item()
    return reg_loss


def get_atoms(module):
    atoms = []
    n_components = len(module.convs)
    for indx in range(n_components):
        atoms.append(module.convs[indx](module.maps[indx]))
    return atoms


def conv_transform(im: torch.Tensor, atoms: list, comp_indx: list):
    x = 0
    for indx in comp_indx:
        x += atoms[indx]
    return x


def get_recon(module):
    atoms = get_atoms(module)
    recon = 0
    for atom in atoms:
        recon += atom
    return recon


def get_residual(im: torch.Tensor, module):
    recon = get_recon(module)
    return im - recon


def evaluate_mods(inputs, mods):
    x = deepcopy(inputs)
    for mod in mods:
        x = mod(x)
    return x
