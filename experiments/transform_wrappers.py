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
plt.style.use('dark_background')
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
        # unfortunately we don't have automatic broadcasting yet
        return (x - self.mean) / self.std

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)
    
class Net_with_transform(nn.Module):
    '''Prepends transformation onto network (with optional normalizaiton after the transform)
    '''
    def __init__(self, model, transform, norm=None, reshape=None):
        '''
        Params
        ------
        norm: Norm_Layer
        '''
        super(Net_with_transform, self).__init__()
        self.transform = transform
        self.norm = norm
        self.reshape = reshape
        self.model = model

    def forward(self, x):
        '''
        Params
        ------
        x: torch.Tensor
            (batch_size, H, W)
        '''
#         print('forwarding', x.shape)
#         x = torch.ifft(x, signal_ndim=2)
        x = self.transform(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.reshape is not None:
            x = self.reshape(x)
        print('post transform', x.shape)
        
        # should be 4d before inputting to the model
        if x.ndim == 2:
            s = x.shape
            x = x.reshape(s[0], 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        print('pre model', x.shape)
        x = self.model(x)
        return x