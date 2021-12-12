import os

import numpy as np

opj = os.path.join
import itertools
from awave.utils.scheduling import run_serial

DIR_FILE = os.path.dirname(os.path.realpath(__file__))  # directory of the config file

if __name__ == '__main__':
    params_to_vary = {
        'seed': [i for i in range(1,11)],
        'subsample': [1],
        'wave': ['db5'],
        'J': [4],
        'mode': ['zero'],
        'init_factor': [1],
        'noise_factor': [0],
        'const_factor': [0],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [100],
        'attr_methods': ['Saliency'],
        'lamL1wave': [0.02],
        'lamL1attr': [0.00368],
        'dirname': ['db5_saliency_subsample'],
#         'warm_start': [False]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals))  # list of tuples

    # iterate
    run_serial(ks, param_combinations, path=opj(DIR_FILE, "ex_cosmology.py"))
