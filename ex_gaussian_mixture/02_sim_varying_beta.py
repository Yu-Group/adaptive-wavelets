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
        'seed': range(10, 15),
        'hidden_dim': range(6, 10),
        'eps': [0.01],
        'beta': np.round(np.linspace(5e-4, 5, 40), 5),
        'attr': [0],
        'dirname': ['vary_beta0']
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