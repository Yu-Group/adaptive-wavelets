import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import sys
from tqdm import tqdm
from functools import partial
import acd
from copy import deepcopy
sys.path.append('..')
sys.path.append('../..')
from os.path import join as oj
sys.path.append('../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from numpy.fft import *
from torch import nn
from random import randint
from style import *
from captum.attr import *
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF
from transform_wrappers import *
import visualize as viz
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
import pandas as pd
from models import FNN
import attributions
import os
device = 'cuda'


class p:
    '''Parameters for simulation
    '''
    n = 50000
    p = 100
    idx_knockout = 12
    window = 0
    n_test = 1000
    out_dir = '/scratch/users/vision/data/cosmo/sim'
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

    def _str(self):
        vals = vars(p)
        return 'n=' + str(vals['n']) + '_p=' + str(vals['p']) + '_knockout=' + str(vals['idx_knockout']) + 'pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}

class s:
    '''Parameters to save
    '''
    results = None
    acc_test = None
    net = None
    transform = None
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}

    
# seed
np.random.seed(13)
torch.manual_seed(13)

# generate data
X = np.random.randn(p.n, p.p).astype(np.float32)
X_t = torch.rfft(torch.Tensor(X), signal_ndim=1)

# define y
band = X_t[:, p.idx_knockout - p.window: p.idx_knockout + p.window + 1]
band_mag = torch.pow(band[..., 0]**2 + band[..., 1]**2, 0.5)
band_mag_mean = torch.mean(band_mag, axis=1)
thresh = np.nanpercentile(band_mag_mean, 50)
y = (band_mag_mean > thresh).cpu().detach().numpy().astype(np.int)

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# fit model
net = NeuralNetClassifier(
    FNN(p=p.p),
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    train_split=None,
)
net.fit(X_train, y_train)
transform = modularize(lambda x: torch.irfft(x, signal_ndim=1)[:, :-1])
mt = Net_with_transform(model=net.module_, transform=transform).to('cpu')
s.acc_test = net.score(X_test, y_test)
s.net = net.module_
s.transform = 'irfft'

# example look at attributions for one example
# x_torch = torch.Tensor(X_test[0].reshape(1, -1))
# x_t = torch.rfft(x_torch, signal_ndim=1).squeeze()
# results_individual = attributions.get_attributions(x_t, mt)

# calculate scores
print('calculating scores...')
results = []
for i in tqdm(range(p.n_test)):
    x_torch = torch.Tensor(X_test[i].reshape(1, -1))
    x_t = torch.rfft(x_torch, signal_ndim=1).squeeze()
    results.append(attributions.get_attributions(x_t, mt))
s.results = pd.DataFrame(results)


# save
os.makedirs(p.out_dir, exist_ok=True)
results = {**p._dict(p), **s._dict(s)}
pkl.dump(results, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))