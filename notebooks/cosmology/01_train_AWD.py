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
        'mode': ['zero'],
        'init_factor': [1],
        'noise_factor': [0],
        'const_factor': [0],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [50],
        'attr_methods': ['Saliency'],
        'lamL1wave': [0.005, 0.01, 0.02],
        'lamL1attr': np.round([0] + list(np.geomspace(0.001, 0.05, 10)), 5),
        'dirname': ['db5_saliency_warmstart_seed=1_new'],
        'warm_start': [True]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations, path=opj(DIR_FILE, "ex_cosmology.py"))
    
