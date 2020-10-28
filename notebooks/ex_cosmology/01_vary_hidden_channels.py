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
        'num_epochs': [100],
        'seed': [13],
        'h_channels': np.repeat(np.arange(1,7), 10),
        'dirname': ['vary_h_channels_seed=13']
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations)