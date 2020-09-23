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
        'num_epochs': [50],
        'seed': [15],
        'hidden_dim': [12],
        'beta': [0, 0.005, 0.01, 0.05, 0.1, 0.5],
        'mu': [0],
        'lamPT': np.round(np.geomspace(1, 1000, 100), 5),
        'lamCI': [0],
        'dirname': ['vary_lamPT_beta_warmstart_seed=15'],
        'warm_start': ['lamPT'],
        'seq_init': [1]
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