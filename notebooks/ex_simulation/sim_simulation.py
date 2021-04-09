import numpy as np
import torch
import torch.nn as nn
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from random import randint
from copy import deepcopy
import pickle as pkl
import argparse
# adaptive-wavelets modules
sys.path.append('../../src')
sys.path.append('../../../src')
sys.path.append('../../src/adaptive_wavelets')
sys.path.append('../../../src/adaptive_wavelets')
from losses import get_loss_f
from train import Trainer
from evaluate import Validator
from transform1d import DWT1d
from utils import get_1dfilts
from wave_attributions import Attributer


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    n = 5000
    d = 64
    scale_knockout = 3
    idx_knockout = 8
    window = 1
    
    # parameters for initialization
    wave = 'db5'
    J = 4
    init_factor = 1
    noise_factor = 0.1  
    
    # parameters for training
    batch_size = 100
    lr = 0.01
    num_epochs = 10
    attr_methods = 'Saliency'
    lamlSum = 1
    lamhSum = 1
    lamL2norm = 1 
    lamCMF = 1
    lamConv = 1
    lamL1wave = 0.1
    lamL1attr = 1     
    target = 0
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/adaptive-wavelets/notebooks/ex_simulation/results" 
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 10)])
    warm_start = None    

    def _str(self):
        vals = vars(p)
        return 'wave=' + str(vals['wave']) + '_lamL1wave=' + str(vals['lamL1wave']) + '_lamL1attr=' + str(vals['lamL1attr']) \
                + '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
class s:
    '''Parameters to save
    '''
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
  