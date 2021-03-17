import numpy as np
import os,sys
sys.path.append('..')
import itertools
from scheduling import run_serial, run_parallel

if __name__ == '__main__':
    
    params_to_vary = {
        'seed': [100],
        'wave': ['db5'],
        'J': [4],
        'init_factor': [1],
        'noise_factor': [0.05],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [10],
        'attr_methods': ['Saliency'],
        'lamL1wave': [0],
        'lamL1attr': np.round(np.geomspace(0.01, 100, 50), 5),
        'dirname': ['db5_saliency_warmstart_lamL1wave=0_seed=100'],
        'warm_start': [True]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations)