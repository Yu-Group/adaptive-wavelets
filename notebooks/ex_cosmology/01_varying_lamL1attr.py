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
        'num_epochs': [100],
        'seed': [100],
        'init_level': [1],
        'noise_level': [0.1],
        'attr_methods': ['Saliency'],
        'lamL1attr': np.round(np.geomspace(1, 100, 20), 5),
        'lr': [0.01],
        'dirname': ['vary_lamL1attr_seeds_initialized']
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations)