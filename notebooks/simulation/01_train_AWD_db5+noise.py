import numpy as np
import os,sys

opj = os.path.join
import itertools
from awd.scheduling import run_serial, run_parallel
DIR_FILE = os.path.dirname(os.path.realpath(__file__)) # directory of the config file

if __name__ == '__main__':
    
    params_to_vary = {
        'seed': [1],
        'wave': ['db5'],
        'J': [4],
        'init_factor': [1],
        'noise_factor': [0.3],
        'const_factor': [0],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [50],
        'attr_methods': ['Saliency'],
        'lamL1wave': np.round(list(np.geomspace(0.00001, 0.0001, 5)), 5),
        'lamL1attr': np.round([0] + list(np.geomspace(0.00001, 50, 20)), 5),
        'dirname': ['db5_saliency_warmstart_seed=1'],
        'warm_start': [True]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations, path=opj(DIR_FILE, "ex_simulation.py"))
