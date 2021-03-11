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
# adaptive-wavelets modules
sys.path.append('../../src')
sys.path.append('../../src/adaptive_wavelets')
from losses import get_loss_f
from train import Trainer, Validator
from transform1d import DWT1d
from utils import get_1dfilts
from wave_attributions import Attributer


parser = argparse.ArgumentParser(description='Biology Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--init_level', type=float, default=1, metavar='N',
                    help='initialization parameter')
parser.add_argument('--noise_level', type=float, default=0.1, metavar='N',
                    help='initialization parameter')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.00001,
                   help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--attr_methods', type=str, default='InputXGradient',
                   help='type of attribution methods to penalize')
parser.add_argument('--lamSum', type=float, default=0,
                   help='weight of sum of lowpass filter')
parser.add_argument('--lamL2norm', type=float, default=0,
                   help='weight of L2norm of lowpass filter')
parser.add_argument('--lamCMF', type=float, default=0,
                   help='weight of CMF condition')
parser.add_argument('--lamL1wave', type=float, default=0,
                   help='weight of the l1-norm of wavelet coeffs')
parser.add_argument('--lamL1attr', type=float, default=0,
                   help='weight of the l1-norm of attributions')
parser.add_argument('--target', type=int, default=0,
                   help='target index to calc interp score')
parser.add_argument('--dirname', default='vary',
                   help='name of directory')


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    data_path = "../../src/dsets/biology/data"
    
    # parameters for initialization
    init_level = 1
    noise_level = 0.1    
    
    # parameters for training
    batch_size = 100
    lr = 0.00001
    num_epochs = 1000
    attr_methods = 'InputXGradient'
    lamSum = 10 
    lamL2norm = 10 
    lamCMF = 10
    lamL1wave = 0.1
    lamL1attr = 1     
    target = 0
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/adaptive-wavelets/notebooks/ex_biology/results" 
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 10)])

    def _str(self):
        vals = vars(p)
        return 'lamSum=' + str(vals['lamSum']) + '_lamL2norm=' + str(vals['lamL2norm']) + '_lamCMF=' + str(vals['lamCMF']) + \
                '_lamL1wave=' + str(vals['lamL1wave']) + '_lamL1attr=' + str(vals['lamL1attr']) + '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']
    
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

if __name__ == '__main__':
    
    args = parser.parse_args()
    for arg in vars(args):
        setattr(p, arg, getattr(args, arg))
    
    # create dir
    out_dir = opj(p.out_dir, p.dirname)
    os.makedirs(out_dir, exist_ok=True)        

    # get dataloader and model
    (train_loader, test_loader), model = load_dataloader_and_pretrained_model(p, img_size=256)
    
    # prepare model
    wt_orig = Wavelet_Transform(wt_type='DWT', wave='db3', mode='symmetric', device='cuda', J=5)
    
    # seed
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)
    # initialize wavelet filters
    wt = initialize_filters(wt_orig, init_level=p.init_level, noise_level=p.noise_level)
    
    # train
    params = list(wt.xfm.parameters()) + list(wt.ifm.parameters())
    optimizer = torch.optim.Adam(params, lr=p.lr)
    loss_f = get_loss_f(lamL1attr=p.lamL1attr)
    trainer = Trainer(model, wt, Attributer, optimizer, loss_f, attr_methods=p.attr_methods, device=device)
    trainer(train_loader, epochs=p.num_epochs)    

    # calculate losses
    print('calculating losses and metric...')    
    loss_v = get_loss_f(lamL1attr=1)
    validator = Validator(model, wt, Attributer, loss_v, attr_methods=p.attr_methods, device=device)
    _, rec_loss, L1attr_loss = validator(test_loader)
    s.train_losses = trainer.train_losses
    s.val_rec_loss = rec_loss
    s.val_L1attr_loss = L1attr_loss
    s.net = wt
    
    # save
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, p._str(p) + '.pkl'), 'wb'))    
    torch.save(wt.state_dict(), opj(out_dir, p._str(p) + '.pth')) 