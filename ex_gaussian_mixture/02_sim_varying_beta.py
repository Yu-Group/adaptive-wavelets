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
import scheduling

if __name__ == '__main__':
    partition = 'low'
    params_to_vary = {
        'out_dir': ["/scratch/users/vision/chandan/disentanglement/gaussians/vary_beta"],
        'num_epochs': [100],
        'seed': range(10, 15),
        'hidden_dim': range(6, 10),
        'eps': [0.01],
        'beta': np.round(np.linspace(5e-4, 5, 40), 5),
        'attr': [0],
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples
    
    scheduling.run_serial(ks, param_combinations)
#     scheduling.run_parallel(ks, param_combinations, partition=partition)