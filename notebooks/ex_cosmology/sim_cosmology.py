import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from random import randint
from copy import deepcopy
import pickle as pkl
import argparse
from models import load_model

sys.path.append('../../src/dsets/cosmology')
from dset import get_dataloader


parser = argparse.ArgumentParser(description='Cosmology Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dirname', default='vary',
                   help='name of directory')


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    data_path = "../../src/dsets/cosmology/data"
    
    # parameters for model architecture
    img_size = (1, 256, 256)
    
    # parameters for training
    batch_size = 64
    lr = 1*1e-4
    num_epochs = 50
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/local-vae/notebooks/ex_cosmology/results" 
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 10)])

    def _str(self):
        vals = vars(p)
        return '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']
    
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
def load_dataloader_and_pretrained_model(p, img_size=256, split_train_test=True):
    """A generic data loader
    """
    data_loader = get_dataloader(p.data_path, 
                                 img_size=img_size,
                                 split_train_test=split_train_test,
                                 batch_size=p.batch_size) 
    model = load_model(model_name='resnet18', device=device, data_path=p.data_path)
    model = model.eval()
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False    

    return data_loader, model
