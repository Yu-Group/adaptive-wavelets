import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
from copy import deepcopy
import itertools
sys.path.append('..')
from scheduling import run_serial, run_parallel

if __name__ == '__main__':
    
    params_to_vary = {
        'num_epochs': [50],
        'seed': [13],
        'hidden_dim': [12],
        'beta': [0],
        'mu': [0],
        'lamPT': [0],
        'lamCI': [0],
        'lamH': [0.01, 0.05, 0.1, 0.5, 1],
        'lamSP': np.round(np.geomspace(1e-4, 2, 30), 5),
        'dirname': ['vary_lamSP_lamH_seed=13'],
        'warm_start': ['lamSP'], 
        'seq_init': [1e-4]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations)