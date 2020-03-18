import numpy as np
import torch
import random
import os, sys
opj = os.path.join
import acd
from copy import deepcopy
sys.path.append('..')
sys.path.append('../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from torch import nn
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF
import transform_wrappers
from captum.attr import (
    InputXGradient,
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
sys.path.append('../..')
from acd_wooseok.acd.scores import cd

def get_basis_and_weight(nmf, data_dict):
    # nmf dictionary
    D = nmf.components_.astype(np.float32)
    # nmf transform
    W = nmf.transform(data_dict['data']).astype(np.float32)
    W_test = nmf.transform(data_dict['data_t']).astype(np.float32)
    return D, W, W_test


def get_subset_indices(W, W_test, basis_indx):
    if 'list' not in str(type(basis_indx)):
        basis_indx = [basis_indx]
    indx = np.array([], dtype='int64')
    indx_t = np.array([], dtype='int64')
    for i in basis_indx:
        indx = np.union1d(indx, np.argwhere(W[:,i] > 0).flatten())
        indx_t = np.union1d(indx_t, np.argwhere(W_test[:,i] > 0).flatten())
    return indx, indx_t


def comp_im_parts(D, W, W_test, basis_indx):
    tiles = np.zeros((1,D.shape[0]), dtype=np.float32)
    tiles[:,basis_indx] = 1

    # nmf
    im_parts = (W*tiles) @ D
    im_parts_t = (W_test*tiles) @ D
    return im_parts, im_parts_t


def get_interaction_labels(W, W_test, basis_indx):
    label0 = np.argwhere((W[:,basis_indx] <= W[:,basis_indx].mean(axis=0)).sum(axis=1) == 1).flatten()
    label1 = np.setdiff1d(np.arange(W.shape[0]), label0)
    label0_t = np.argwhere((W_test[:,basis_indx] <= W[:,basis_indx].mean(axis=0)).sum(axis=1) == 1).flatten()
    label1_t = np.setdiff1d(np.arange(W_test.shape[0]), label0_t)
    return label0, label1, label0_t, label1_t


# dataloader on the subset
def load_data_on_subset(data_dict: dict,
                        train_batch_size,
                        test_batch_size,
                        device,
                        return_interp_loader=True,
                        return_indices=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = dset.MyDataset(data_dict['data'], data_dict['targets'],
                            transform=transformer, return_indices=return_indices)
    test_set = dset.MyDataset(data_dict['data_t'], data_dict['targets_t'],
                            transform=transformer, return_indices=return_indices)
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=train_batch_size,
                                              shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False, **kwargs)
    if return_interp_loader:
        num = len(data_dict['targets_t']) - data_dict['targets_t'].sum()
        interp_set = torch.utils.data.Subset(test_set, np.arange(num))
        interp_loader = torch.utils.data.DataLoader(
                        interp_set, batch_size=test_batch_size, shuffle=False, **kwargs)
        return (train_loader, test_loader, interp_loader)
    return (train_loader, test_loader)


# dataloader for full images vs images with basis removed
def dataloader_nmf_knockout(args, nmf, basis_indx=0,
                            return_interp_loader=True,
                            return_indices=True,
                            task_type='remove_one_basis'):
    if args.cuda:
        device = 'cuda'
    # load dataloaders
    train_loader, test_loader = dset.load_data(args.batch_size,
                                          args.test_batch_size,
                                          device,
                                          return_indices=return_indices)

    # load data dictionary
    data_dict = dset.load_mnist_arrays(train_loader, test_loader)
    # nmf
    D, W, W_test = get_basis_and_weight(nmf, data_dict)
    # select data subset indices
    indx, indx_t = get_subset_indices(W, W_test, basis_indx)
    W = W[indx]
    W_test = W_test[indx_t]
    # residual
    res = torch.Tensor(data_dict['data_t'][indx_t] - W_test @ D)

    if task_type == 'remove_one_basis':
        im_parts, im_parts_t = comp_im_parts(D, W, W_test, basis_indx)
        # define new dictionary
        data_dict = {
            'data': np.r_[data_dict['data'][indx], data_dict['data'][indx] - im_parts],
            'data_t': np.r_[data_dict['data_t'][indx_t], data_dict['data_t'][indx_t] - im_parts_t],
            'targets': np.r_[np.zeros(len(indx), dtype=int), np.ones(len(indx), dtype=int)],
            'targets_t': np.r_[np.zeros(len(indx_t), dtype=int), np.ones(len(indx_t), dtype=int)],
        }
    elif task_type == 'interaction':
        label0, label1, label0_t, label1_t = get_interaction_labels(W, W_test, basis_indx)
        # define new dictionary
        data_dict = {
            'data': np.r_[data_dict['data'][indx][label0], data_dict['data'][indx][label1]],
            'data_t': np.r_[data_dict['data_t'][indx_t][label0_t], data_dict['data_t'][indx_t][label1_t]],
            'targets': np.r_[np.zeros(len(label0), dtype=int), np.ones(len(label1), dtype=int)],
            'targets_t': np.r_[np.zeros(len(label0_t), dtype=int), np.ones(len(label1_t), dtype=int)],
        }

    # residual
    data_dict['W_test_t'] = torch.Tensor(W_test)
    data_dict['res'] = res

    data_loaders = load_data_on_subset(data_dict,
                                       args.batch_size,
                                       args.test_batch_size,
                                       device,
                                       return_interp_loader=return_interp_loader,
                                       return_indices=return_indices)
    return data_loaders, data_dict


def comp_grad_scores(model, nmf, interp_loader, data_dict, grad_mode='exact', device='cuda'):
    # network with transform augmented
    transform = transform_wrappers.lay_from_w(nmf.components_)
    norm = transform_wrappers.NormLayer(mu=0.1307, std=0.3081)
    reshape = transform_wrappers.ReshapeLayer(shape=(1, 28, 28))
    net = transform_wrappers.Net_with_transform(model,
                                                transform=transform,
                                                norm=norm,
                                                reshape=reshape,
                                                use_logits=True).to(device)
    if grad_mode == 'exact':
        net.n_components = nmf.n_components

    # interp modules
    gradient_shap = GradientShap(net)
    ig = IntegratedGradients(net)
    saliency = Saliency(net)
    input_x_gradient = InputXGradient(net)

    # store results
    results = {
        'gradient_shap': [],
        'ig': [],
        'saliency': [],
        'input_x_gradient': []
    }
    n_components_ = nmf.n_components_
    for batch_indx, (data, target, data_indx) in enumerate(interp_loader):
        if grad_mode == 'exact':
            x_t = torch.cat((data_dict['W_test_t'][data_indx], data_dict['res'][data_indx]), dim=1)
        else:
            x_t = data_dict['W_test_t'][data_indx]
        x_t = x_t.to(device).requires_grad_(True)
        # comp gradient
        baselines = torch.zeros_like(x_t)
        results['gradient_shap'].append(gradient_shap.attribute(x_t, baselines=baselines,
                                                                target=0).cpu().detach().numpy()[:,:n_components_])
        results['ig'].append(ig.attribute(x_t, target=0).cpu().detach().numpy()[:,:n_components_])
        results['saliency'].append(saliency.attribute(x_t, target=0, abs=False).cpu().detach().numpy()[:,:n_components_])
        results['input_x_gradient'].append(input_x_gradient.attribute(x_t, target=0).cpu().detach().numpy()[:,:n_components_])

        print('\r batch index: {}'.format(batch_indx), end='')

    results['gradient_shap'] = np.vstack(results['gradient_shap'])
    results['ig'] = np.vstack(results['ig'])
    results['saliency'] = np.vstack(results['saliency'])
    results['input_x_gradient'] = np.vstack(results['input_x_gradient'])
    return results


def comp_cd_scores(model, nmf, interp_loader, data_dict, cd_mode='cd', device='cuda'):
    # nmf transform layers
    nmf_transformer = transform_wrappers.TransformLayers(nmf.components_).to(device)

    # convert nmf weight to tensor
    sweep_dim = 1
    tiles = torch.Tensor(acd.tiling_2d.gen_tiles(data_dict['W_test_t'][0:1], fill=0, method='cd', sweep_dim=sweep_dim)).to(device)
    data_dict['W_test_t'] = data_dict['W_test_t'].to(device)

    # store results
    results = {
        'cd': []
    }
    for batch_indx, (data, target, data_indx) in enumerate(interp_loader):
        # loop over nmf basis
        scores_cd = []
        for basis_indx in range(nmf.n_components):
            im_parts = nmf_transformer(data_dict['W_test_t'][data_indx]*tiles[basis_indx])
            scores_cd.append(cd.cd(data, model, mask=None, model_type=None, device='cuda', transform=None,
                                     relevant=im_parts)[0].data.cpu().numpy()[:,0])

            print('\r batch index: {} [basis component index: {}]'.format(batch_indx, basis_indx), end='')
        scores_cd = np.vstack(scores_cd).T
        results['cd'].append(scores_cd)
    results['cd'] = np.vstack(results['cd'])
    return results