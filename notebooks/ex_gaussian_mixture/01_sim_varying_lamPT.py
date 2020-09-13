import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
import acd
from copy import deepcopy
import itertools


if __name__ == '__main__':
    
    params_to_vary = {
        'num_epochs': [100],
        'seed': [13],
        'hidden_dim': [12],
        'beta': [0],
        'mu': [0, 0.01, 0.1],
        'lamPT': np.round(np.linspace(0.1, 100, 30), 5),
        'lamCI': [0],
        'dirname': ['vary_lamPT_mu']
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples
    
    # iterate
    for i in range(len(param_combinations)):
        param_str = 'python sim_gaussian_mixture.py '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print('running: ' + param_str + '({}/{})'.format(i, len(param_combinations)))
        os.system(param_str)