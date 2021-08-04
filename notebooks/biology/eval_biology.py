import numpy as np
import torch
from copy import deepcopy
import pickle as pkl
import pandas as pd
import os,sys
opj = os.path.join

sys.path.append('../../src/adaptive_wavelets')
sys.path.append('../..')
from src import adaptive_wavelets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tensors(data_loader):
    """Given dataloader return inputs and labels in torch.Tensor
    """
    inputs, labels = data_loader.dataset.tensors
    X = deepcopy(inputs)
    y = deepcopy(labels)    
    return (X, y)


def max_fun(X, sgn="abs", m=1):
    """Given an array X return maximum values across columns for every row
    """
    if sgn == "abs":
        Y = abs(X)
    elif sgn == "neg":
        Y = -X
    elif sgn == "pos":
        Y = X
    else:
        print('no such sign supported')
    id_s = np.argsort(Y, axis=1)[:,::-1]
    index = id_s[:,:m]
    return np.take_along_axis(X, index, axis=1)


def max_transformer(w_transform, 
                    train_loader, 
                    test_loader,
                    sgn="abs", 
                    m=1):
    """Compute maximum features of wavelet representations across all scales 
    """
    w_transform = w_transform.to('cpu')
    J = w_transform.J
    
    # transform train data
    (Xs, y) = get_tensors(train_loader)
    X = []
    data_t = w_transform(Xs)
    for j in range(J+1):
        d = data_t[j].detach().squeeze().numpy()
        X.append(max_fun(d, sgn=sgn, m=m))
    X = np.hstack(X)
    y = y.detach().squeeze().numpy()
    
    # transform test data
    (Xs_test, y_test) = get_tensors(test_loader)
    X_test = []
    data_t = w_transform(Xs_test)
    for j in range(J+1):
        d = data_t[j].detach().squeeze().numpy()
        X_test.append(max_fun(d, sgn=sgn, m=m))
    X_test = np.hstack(X_test)
    y_test = y_test.detach().squeeze().numpy()    
    
    return (X, y), (X_test, y_test)  


def load_results(dirs, wave='db5', include_interp_loss=True):
    """load results for analysis
    """
    results = []
    models = []
    for i in range(len(dirs)):
        # load results
        out_dir = opj("/home/ubuntu/adaptive-wavelets/notebooks/biology/results", dirs[i])
        fnames = sorted(os.listdir(out_dir))

        results_list = []
        models_list = []
        for fname in fnames:
            if include_interp_loss:
                if fname[-3:] == 'pkl':
                    results_list.append(pkl.load(open(opj(out_dir, fname), 'rb')))
                if fname[-3:] == 'pth':
                    wt = adaptive_wavelets.DWT1d(wave=wave, mode='zero', J=4, init_factor=1, noise_factor=0.0).to(device)
                    wt.load_state_dict(torch.load(opj(out_dir, fname)))
                    models_list.append(wt)
            else:
                if "lamL1attr=0.0_" in fname:
                    if fname[-3:] == 'pkl':
                        results_list.append(pkl.load(open(opj(out_dir, fname), 'rb')))
                    if fname[-3:] == 'pth':
                        wt = adaptive_wavelets.DWT1d(wave=wave, mode='zero', J=4, init_factor=1, noise_factor=0.0).to(device)
                        wt.load_state_dict(torch.load(opj(out_dir, fname)))
                        models_list.append(wt)                
                
        results.append(pd.DataFrame(results_list))
        models.append(models_list)

    dics = []
    for i in range(len(dirs)):
        # define indexes
        res = results[i]
        mos = models[i]
        lamL1wave = np.array(res['lamL1wave'])
        lamL1attr = np.array(res['lamL1attr'])
        lamL1wave_grid = np.unique(lamL1wave)
        lamL1attr_grid = np.unique(lamL1attr)
        R = len(lamL1wave_grid)
        C = len(lamL1attr_grid)

        # collect results
        dic = {'phi':{},
               'psi':{},
               'wt': {},
               'x': {},
               'lamL1wave': {},
               'lamL1attr': {},
               'index': {}}

        for r in range(R):
            for c in range(C):
                loc = (lamL1wave == lamL1wave_grid[r]) & (lamL1attr == lamL1attr_grid[c])
                if loc.sum() == 1: 
                    loc = np.argwhere(loc).flatten()[0]
                    dic['index'][(r,c)] = loc
                    wt = mos[loc]
                    phi, psi, x = adaptive_wavelets.get_wavefun(wt)

                    dic['wt'][(r,c)] = wt
                    dic['phi'][(r,c)] = phi
                    dic['psi'][(r,c)] = psi  
                    dic['x'][(r,c)] = x
                    dic['lamL1wave'][(r,c)] = lamL1wave_grid[r]
                    dic['lamL1attr'][(r,c)] = lamL1attr_grid[c]
        dics.append(dic)    
        
    return dics, results, models