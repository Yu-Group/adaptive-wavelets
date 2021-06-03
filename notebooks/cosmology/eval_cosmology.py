import numpy as np
import torch
import random
from copy import deepcopy
import pickle as pkl
import pandas as pd
import os,sys
opj = os.path.join
from transform2d import DWT2d
from utils import get_2dfilts, get_wavefun
from peak_counting import rmse
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_results(dirs, wave='db5'):
    """load results for analysis
    """
    results = []
    models = []
    for i in range(len(dirs)):
        # load results
        out_dir = opj("/home/ubuntu/adaptive-wavelets/notebooks/cosmology/results", dirs[i])
        fnames = sorted(os.listdir(out_dir))

        results_list = []
        models_list = []
        for fname in fnames:
            if fname[-3:] == 'pkl':
                results_list.append(pkl.load(open(opj(out_dir, fname), 'rb')))
            if fname[-3:] == 'pth':
                wt = DWT2d(wave=wave, mode='zero', J=4, init_factor=1, noise_factor=0.0).to(device)
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
        dic = {'psi':{},
               'wt': {},
               'x': {},
               'lamL1wave': {},
               'lamL1attr': {},
               'index': {}}

        for r in range(R):
            for c in range(C):
                if lamL1attr_grid[c] <= 0.1 and lamL1wave_grid[r] <= 0.05:
                    loc = (lamL1wave == lamL1wave_grid[r]) & (lamL1attr == lamL1attr_grid[c])
                    if loc.sum() == 1: 
                        loc = np.argwhere(loc).flatten()[0]
                        dic['index'][(r,c)] = loc
                        wt = mos[loc]
                        _, psi, x = get_wavefun(wt)

                        dic['wt'][(r,c)] = wt
                        dic['psi'][(r,c)] = psi  
                        dic['x'][(r,c)] = x
                        dic['lamL1wave'][(r,c)] = lamL1wave_grid[r]
                        dic['lamL1attr'][(r,c)] = lamL1attr_grid[c]
        dics.append(dic)    
        
    return dics, results, models


def rmse_bootstrap(y, y_pred, target=1, m=10000):
    """Return bootstrap mean and std error."""
    np.random.seed(1)
    e = []
    for i in range(m):
        idx = np.arange(len(y))
        sel = np.random.choice(idx, len(idx), replace=True)
        e.append(rmse(y[sel],y_pred[sel], target))
    return rmse(y, y_pred, target), np.std(e)


def extract_patches(h, g, centering=True):
    """Given 1-d filters h, g, extract 3x3 LL,LH,HL,HH filters with largest variation
    """
    hc = h - h.mean()
    var = []
    for left in range(len(h)-3):
        v = torch.sum((hc[left:left+3])**2)
        var.append(v)
    var = np.array(var)
    h_small = h[np.argmax(var):np.argmax(var)+3]
    
    gc = g - g.mean()
    var = []
    for left in range(len(g)-3):
        v = torch.sum((gc[left:left+3])**2)
        var.append(v)
    var = np.array(var)
    g_small = g[np.argmax(var):np.argmax(var)+3]
    
    ll = h_small.unsqueeze(0)*h_small.unsqueeze(1)
    lh = h_small.unsqueeze(0)*g_small.unsqueeze(1)
    hl = g_small.unsqueeze(0)*h_small.unsqueeze(1)
    hh = g_small.unsqueeze(0)*g_small.unsqueeze(1)
    
    if centering:
        lh -= lh.mean()
        hl -= hl.mean()
        hh -= hh.mean()
    
    return [ll, lh, hl, hh]