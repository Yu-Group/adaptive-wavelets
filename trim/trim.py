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
from util import *
from numpy.fft import *
from torch import nn
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF


class TrimModel(nn.Module):
    '''Prepends transformation onto network (with optional normalizaiton after the transform)

    Params
    ------
    model: nn.Module
        model after all the transformations
    inv_transform: nn.Module
        the inverse transform
    norm: nn.Module (Norm_Layer)
        normalization to apply after the inverse transform
    reshape: nn.Module
        reshape to apply after the normalization
    use_residuals: bool, optional
        whether or not to apply the residuals after the transformation 
        (for transformations which are not perfectly invertible)
    use_logits: bool, optional
        whether to use the logits (if the model has them) or the forward function
    n_components: int
        right now this setup is kind of weird - if you want to pass a residual
        pass x as a 1d vector whose last entries contain the residual [x, residual]
    '''
    def __init__(self, model, inv_transform, norm=None, reshape=None, 
                 use_residuals=False, use_logits=False, n_components=None):
        super(TrimModel, self).__init__()
        self.inv_transform = inv_transform
        self.norm = norm
        self.reshape = reshape
        self.model = model
        self.use_residuals = use_residuals
        self.use_logits = use_logits
        self.n_components = n_components

    def forward(self, s, x_orig=None):
        '''
        Params
        ------
        s: torch.Tensor
            This should be the input in the transformed space which we want to interpret
            (batch_size, C, H, W) for images
            (batch_size, C, seq_length) for audio
        '''
        # untransform the input
        x = self.inv_transform(s)
        
        # take residuals into account
        if self.use_residuals:
            assert x is not None, "if using residuals, must also pass untransformed original image!"
            with torch.no_grad():
                res = x_orig - x
            x = x + res
        
        # normalize
        if self.norm is not None:
            x = self.norm(x)
            
        # reshape
        if self.reshape is not None:
            x = self.reshape(x)
            
        # pass through the main model
        if self.use_logits:
            x = self.model.logits(x)
        else:
            x = self.model.forward(x)
        return x
    
def lay_from_w(D: np.ndarray):
    '''Creates a linear layer given a weight matrix
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
        super(NormLayer, self).__init__()
        self.mean = mu
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

def modularize(f):
    '''Turns any function into a torch module
    '''
    class Transform(nn.Module):
        def __init__(self, f):
            super(Transform, self).__init__()
            self.f = f
        def forward(self, x):
            return self.f(x)
    return Transform(f)    
    
class ReshapeLayer(nn.Module):
    '''Returns a torch module which reshapes an input to a desired shape
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