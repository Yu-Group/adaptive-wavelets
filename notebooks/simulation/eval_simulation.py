import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch

opj = os.path.join
from awd.transform1d import DWT1d
from awd.utils import get_wavefun, dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_results(dirs, waves, path):
    """load results for analysis
    """
    results = []
    models = []
    for i in range(len(dirs)):
        # load results
        out_dir = opj(path, dirs[i])
        fnames = sorted(os.listdir(out_dir))

        results_list = []
        models_list = []
        for fname in fnames:
            if fname[-3:] == 'pkl':
                results_list.append(pkl.load(open(opj(out_dir, fname), 'rb')))
            if fname[-3:] == 'pth':
                wt = DWT1d(wave=waves[i], mode='zero', J=4, init_factor=1, noise_factor=0.0).to(device)
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

        # original wavelet
        wt_o = DWT1d(wave='db5', mode='zero', J=4).to(device)
        phi_o, psi_o, x_o = get_wavefun(wt_o)

        # collect results
        dic = {'psi': {},
               'wt': {},
               'x': {},
               'dist': {},
               'lamL1wave': {},
               'lamL1attr': {},
               'index': {}}

        for r in range(R):
            for c in range(C):
                loc = (lamL1wave == lamL1wave_grid[r]) & (lamL1attr == lamL1attr_grid[c])
                if loc.sum() == 1:
                    loc = np.argwhere(loc).flatten()[0]
                    dic['index'][(r, c)] = loc
                    wt = mos[loc]
                    _, psi, x = get_wavefun(wt)
                    d = dist(wt, wt_o)

                    dic['wt'][(r, c)] = wt
                    dic['psi'][(r, c)] = psi
                    dic['x'][(r, c)] = x
                    dic['dist'][(r, c)] = d
                    dic['lamL1wave'][(r, c)] = lamL1wave_grid[r]
                    dic['lamL1attr'][(r, c)] = lamL1attr_grid[c]
        dics.append(dic)

    return dics, results, models
