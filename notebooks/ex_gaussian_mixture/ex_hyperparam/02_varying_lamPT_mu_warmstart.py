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
from scheduling import run_serial, run_parallel

if __name__ == '__main__':
    
    params_to_vary = {
        'num_epochs': [50],
        'seed': [13],
        'hidden_dim': [12],
        'beta': [0],
        'mu': [0, 0.005, 0.01, 0.05, 0.1, 0.5],
        'lamPT': np.round(np.linspace(1, 300, 60), 5),
        'lamCI': [0],
        'dirname': ['vary_lamPT_mu_warmstart'],
        'warm_start': ['lamPT'],
        'seq_init': [1]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations)