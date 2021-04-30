import numpy as np
import os,sys
sys.path.append('../../src')
opj = os.path.join
import itertools
from scheduling import run_serial, run_parallel
DIR_FILE = os.path.dirname(os.path.realpath(__file__)) # directory of the config file

if __name__ == '__main__':
    
    params_to_vary = {
        'seed': [1],
        'wave': ['coif2'],
        'J': [4],
        'init_factor': [1],
        'noise_factor': [0],
        'const_factor': [0],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [100],
        'attr_methods': ['InputXGradient'],
        'lamL1wave': np.round(list(np.geomspace(0.0001, 0.005, 5)), 5),
        'lamL1attr': np.round([0] + list(np.geomspace(0.0001, 10, 20)), 5),
        'dirname': ['coif2_inputxgrad_warmstart_seed=1'],
        'warm_start': [True]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples    
    
    # iterate
    run_serial(ks, param_combinations, path=opj(DIR_FILE, "ex_biology.py"))
