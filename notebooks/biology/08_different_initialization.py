import os

opj = os.path.join
import itertools
from awd.scheduling import run_serial

DIR_FILE = os.path.dirname(os.path.realpath(__file__))  # directory of the config file

if __name__ == '__main__':
    params_to_vary = {
        'seed': [1],
        'wave': ['db5'],
        'J': [4],
        'init_factor': [0],
        'noise_factor': [0.1, 0.2, 0.3, 0.5, 1],
        'const_factor': [0],
        'batch_size': [100],
        'lr': [0.001],
        'num_epochs': [600],
        'attr_methods': ['Saliency'],
        'lamL1wave': [0.001],
        'lamL1attr': [0.01, 0.1, 1.0],
        'dirname': ['random_init_saliency_seed=1'],
        'warm_start': [None]
    }
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals))  # list of tuples

    # iterate
    run_serial(ks, param_combinations, path=opj(DIR_FILE, "ex_biology.py"))
