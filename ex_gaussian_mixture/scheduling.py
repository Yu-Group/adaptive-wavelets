import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
from copy import deepcopy
import itertools
    
def run_serial(ks, param_combinations):    
    '''run each parameter combination in serial
    '''
    for i in range(len(param_combinations)):
        param_str = 'python sim_gaussian_mixture.py '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print('running: ' + param_str + '({}/{})'.format(i, len(param_combinations)))
        os.system(param_str)
        
def run_parallel(ks, param_combinations, partition='low'):    
    '''run each parameter combination in parallel (requires slurmpy package)
    '''
    from slurmpy import Slurm
    s = Slurm("fit_mog", {"partition": partition, "time": "4-0"})
    
    for i in range(len(param_combinations)):
        param_str = 'module load python; python3 sim_gaussian_mixture.py '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print('scheduled: ' + param_str + '({}/{})'.format(i, len(param_combinations)))
        s.run(param_str)
