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
from util import *
from numpy.fft import *
from torch import nn
import random
from random import randint
from style import *
from captum.attr import *
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF, LatentDirichletAllocation
import transform_wrappers
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
    transform = 'nmf' # 'fft', 'nmf', 'lda'
    lr = 0.01 # 0.01 works for nmf, 0.1 works for fft
    data_distr = 'uniform' # 'normal', 'uniform'
    window = 0
    n_test = 500
    num_epochs_train = 12
    num_components = 30
    out_dir = '/scratch/users/vision/data/cosmo/sim/test'
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
    preds = None
    labels = None
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}

    

# define transformations (Tensor -> Tensor)
def get_transforms(X, p):
    if p.transform == 'fft':
        t = lambda x: torch.rfft(x, signal_ndim=1)
        transform_i = transform_wrappers.modularize(lambda x: torch.irfft(x, signal_ndim=1)[:, :-1])
    elif p.transform in ['nmf', 'lda']:
        X = X - np.min(X)
        if p.transform == 'nmf':
            decomp = NMF(n_components=p.num_components)
        else:
            decomp = LatentDirichletAllocation(n_components=p.num_components)

        fname = f'{p.transform}_{p.num_components}_{p.data_distr}.pkl'
        if os.path.exists(fname):
            decomp = pkl.load(open(fname, 'rb'))
        else:
            print('fitting decomp...')
            decomp.fit(X)
            pkl.dump(decomp, open(fname, 'wb'))
        t = lambda x: torch.Tensor(decomp.transform(x))
        transform_i = transform_wrappers.lay_from_w(decomp.components_)
    return X, t, transform_i


# generate data
def define_y(X_t, p):
    if p.transform == 'fft':
        band = X_t[:, p.idx_knockout - p.window: p.idx_knockout + p.window + 1]
        band_mag = torch.pow(band[..., 0]**2 + band[..., 1]**2, 0.5)
        band_mag_mean = torch.mean(band_mag, axis=1)
        thresh = np.nanpercentile(band_mag_mean, 50)
        y = band_mag_mean > thresh
    else:
        thresh = np.nanpercentile(X_t[:, p.idx_knockout], 50)
        y = X_t[:, p.idx_knockout] > thresh
    if 'Tensor' in str(type(y)):
        y = y.cpu().detach().numpy()
    return y.astype(np.int)



if __name__ == '__main__':
    # seed
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)

    if p.data_distr == 'normal':
        X = np.random.randn(p.n, p.p).astype(np.float32)
    elif p.data_distr == 'uniform':
        X = np.random.rand(p.n, p.p).astype(np.float32)

    X, t, transform_i = get_transforms(X, p)

    X_t = t(torch.Tensor(X))
    y = define_y(X_t, p)


    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    # fit model
    net = NeuralNetClassifier(
        FNN(p=p.p),
        max_epochs=p.num_epochs_train,
        lr=p.lr,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        train_split=None,
    )
    net.fit(X_train, y_train)
    mt = transform_wrappers.Net_with_transform(model=net.module_, transform=transform_i).to('cpu')
    s.acc_test = net.score(X_test, y_test)
    print('test acc', s.acc_test)
    s.net = net.module_
    s.preds = net.predict(X_test[:p.n_test])
    s.labels = y[:p.n_test]

    # example look at attributions for one example
    # x_torch = torch.Tensor(X_test[0].reshape(1, -1))
    # x_t = torch.rfft(x_torch, signal_ndim=1).squeeze()
    # results_individual = attributions.get_attributions(x_t, mt)


    # calculate scores
    print('calculating scores...')
    results = []
    for i in tqdm(range(p.n_test)):
        x_torch = torch.Tensor(X_test[i].reshape(1, -1))
        x_t = t(x_torch).squeeze()
        results.append(attributions.get_attributions(x_t, mt))
    s.results = pd.DataFrame(results)


    # save
    os.makedirs(p.out_dir, exist_ok=True)
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))