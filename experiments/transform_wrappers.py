import numpy as np
import matplotlib.pyplot as plt
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.ndimage import gaussian_filter
import sys
from tqdm import tqdm
from functools import partial
import acd
from copy import deepcopy
sys.path.append('..')
from transforms_torch import bandpass_filter
# plt.style.use('dark_background')
sys.path.append('../../dsets/mnist')
import dset
from model import Net
from util import *
from numpy.fft import *
from torch import nn
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF

def lay_from_w(D: np.ndarray):
    '''
    Params
    ------
    D
        weight matrix (in_features, out_features)
    '''
    lay = nn.Linear(in_features=D.shape[0], out_features=D.shape[1], bias=False)
    lay.weight.data = torch.tensor(D.astype(np.float32)).T
    return lay

class NormLayer(nn.Module):
    '''Normalizes images (assumes only 1 channel)
    image = (image - mean) / std
    '''
    def __init__(self, mu=0.1307, std=0.3081):
#         transforms.Normalize((0.1307,), (0.3081,))
        super(NormLayer, self).__init__()
        self.mean = mu
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class ReshapeLayer(nn.Module):
    '''Reshapes input after transformation, before feeding to network

    Params
    ------
    shape: tuple
        shape excluding batch size
    '''
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

class Net_with_transform(nn.Module):
    '''Prepends transformation onto network (with optional normalizaiton after the transform)

    Params
    ------
    model: nn.Module
        model after all the transformations
    transform: nn.Module
        the inverse transform
    norm: nn.Module (Norm_Layer)
        normalization to apply after the inverse transform
    reshape: nn.Module
        reshape to apply after the normalization
    use_logits: bool, optional
        whether to use the logits (if the model has it) or the forward function
    '''
    def __init__(self, model, transform, norm=None, reshape=None, use_logits=False, n_components=None):
        super(Net_with_transform, self).__init__()
        self.transform = transform
        self.norm = norm
        self.reshape = reshape
        self.model = model
        self.use_logits = use_logits
        self.n_components = n_components

    def forward(self, x):
        '''
        Params
        ------
        x: torch.Tensor
            (batch_size, C, H, W) for images
            (batch_size, C, seq_length) for audio
        '''
#         print('forwarding', x.shape)
        if self.n_components is not None:
            res = x[:,self.n_components:]
            x = self.transform(x[:,:self.n_components]) + res
        else:
            x = self.transform(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.reshape is not None:
            x = self.reshape(x)
#         print('post transform', x.shape)

        # should be 4d before inputting to the model
        '''
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        '''

#         print('pre model', x.shape)
        if self.use_logits:
            x = self.model.logits(x)
        else:
            x = self.model.forward(x)
        return x


class TransformLayers(nn.Module):
    def __init__(self, D):
        super(TransformLayers, self).__init__()
        self.transform = lay_from_w(D)
        self.norm_nmf = NormLayer(mu=0.0, std=0.3081)
        self.norm_img = NormLayer(mu=0.1307, std=0.3081)
        self.reshape = ReshapeLayer(shape=(1, 28, 28))

    def forward(self, x):
        '''
        Params
        ------
        x: torch.Tensor
            (batch_size, C, H, W) for images
            (batch_size, C, seq_length) for audio
        '''
#         print('forwarding', x.shape)
        x = self.transform(x)
        x = self.norm_nmf(x)
        x = self.reshape(x)
        return x
        # should be 4d before inputting to the model
        '''
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        '''

    def im_reshape(self, x):
        x = self.norm_img(x)
        x = self.reshape(x)
        return x
