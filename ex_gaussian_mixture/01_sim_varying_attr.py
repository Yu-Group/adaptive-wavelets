import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
from copy import deepcopy
import itertools
import scheduling

if __name__ == '__main__':
    partition = 'low'
    params_to_vary = {
        'out_dir': ["/scratch/users/vision/chandan/disentanglement/gaussians/vary_attr"],
        'num_epochs': [100],
        'seed': range(10, 13),
        'hidden_dim': [6, 10, 25], #range(6, 10),
        'eps': [0.01, 0.05, 0.1],
        'beta': [0],
        'attr': np.round(np.logspace(-5, 2, 6), 5),
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples
    
    # iterate
#     print('num jobs:', len(param_combinations))
#     scheduling.run_serial(ks, param_combinations)
    scheduling.run_parallel(ks, param_combinations, partition=partition)    