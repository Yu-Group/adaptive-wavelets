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

sys.path.append('preprocessing')
import data
import neural_networks

sys.path.append('../../src/dsets/biology')
from dset import get_dataloader


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    data_path = "../../src/dsets/biology/data"
    
    # parameters for training
    batch_size = 100    
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/local-vae/notebooks/ex_biology/results" 
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 10)])

    def _str(self):
        vals = vars(p)
        return 'lamL1attr=' + str(vals['lamL1attr']) + '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
class s:
    '''Parameters to save
    '''
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
# generate data
def load_dataloader_and_pretrained_model(p, batch_size=100):
    """A generic data loader
    """
    data_loader = get_dataloader(p.data_path, 
                                 batch_size=p.batch_size) 
    
    results = pkl.load(open(opj(p.data_path, 'dnn_full_long_normalized_across_track_1_feat.pkl'), 'rb'))
    dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm')
    dnn.model.load_state_dict(results['model_state_dict'])
    m = deepcopy(dnn.model)
#     m = m.eval()
    # freeze layers
    for param in m.parameters():
        param.requires_grad = False  
    model = ReshapeModel(m)

    return data_loader, model


class ReshapeModel(nn.Module):
    def __init__(self, model):
        super(ReshapeModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = x.squeeze()
        return self.model(x)

  

