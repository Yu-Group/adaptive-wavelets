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
sys.path.append('../preprocessing')
import data
import neural_networks
sys.path.append('../../src/dsets/biology')
sys.path.append('../../../src/dsets/biology')
from dset import get_dataloader
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


parser = argparse.ArgumentParser(description='Biology Example')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--wave', type=str, default='db5', help='type of wavelet')
parser.add_argument('--J', type=int, default=4, help='level of resolution')
parser.add_argument('--init_factor', type=float, default=1, metavar='N', help='initialization parameter')
parser.add_argument('--noise_factor', type=float, default=0.1, metavar='N', help='initialization parameter')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--attr_methods', type=str, default='Saliency', help='type of attribution methods to penalize')
parser.add_argument('--lamlSum', type=float, default=1, help='weight of sum of lowpass filter')
parser.add_argument('--lamhSum', type=float, default=1, help='weight of sum of highpass filter')
parser.add_argument('--lamL2norm', type=float, default=1, help='weight of L2norm of lowpass filter')
parser.add_argument('--lamCMF', type=float, default=1, help='weight of CMF condition')
parser.add_argument('--lamConv', type=float, default=1, help='weight of convolution constraint')
parser.add_argument('--lamL1wave', type=float, default=0, help='weight of the l1-norm of wavelet coeffs')
parser.add_argument('--lamL1attr', type=float, default=0, help='weight of the l1-norm of attributions')
parser.add_argument('--target', type=int, default=0, help='target index to calc interp score')
parser.add_argument('--dirname', default='vary', help='name of directory')
parser.add_argument('--warm_start', default=None, help='indicate whether warmstart or not')


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    data_path = "../../../src/dsets/biology/data"
    model_path = "../../../src/dsets/biology/data"
    is_continuous = False
    
    # parameters for initialization
    wave = 'db5'
    J = 4
    init_factor = 1
    noise_factor = 0.1    
    
    # parameters for training
    batch_size = 100
    lr = 0.001
    num_epochs = 100
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
    out_dir = "/home/ubuntu/adaptive-wavelets/notebooks/biology/results" 
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
    
    
# generate data
def load_dataloader_and_pretrained_model(p):
    """A generic data loader
    """
    data_loader = get_dataloader(p.data_path, 
                                 batch_size=p.batch_size,
                                 is_continuous=p.is_continuous) 
    model = load_model(p)
    return data_loader, model


def load_model(p):
    results = pkl.load(open(opj(p.model_path, 'dnn_full_long_normalized_across_track_1_feat.pkl'), 'rb'))
    dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm')
    dnn.model.load_state_dict(results['model_state_dict'])
    m = deepcopy(dnn.model)
    # freeze layers
    for param in m.parameters():
        param.requires_grad = False  
    model = ReshapeModel(m)
    return model


def warm_start(p, out_dir):
    '''load results and initialize model 
    '''
    print('\twarm starting...')
    fnames = sorted(os.listdir(out_dir))
    params = []
    models = []
    if len(fnames) == 0:
        model = DWT1d(wave=p.wave, mode='zero', J=p.J, init_factor=p.init_factor, noise_factor=p.noise_factor).to(device)
    else:
        for fname in fnames:
            if fname[-3:] == 'pkl':
                result = pkl.load(open(opj(out_dir, fname), 'rb'))
                params.append(result['lamL1attr'])
            if fname[-3:] == 'pth':
                m = DWT1d(wave=p.wave, mode='zero', J=p.J, init_factor=p.init_factor, noise_factor=p.noise_factor).to(device)
                m.load_state_dict(torch.load(opj(out_dir, fname)))
                models.append(m)
        max_idx = np.argmax(np.array(params))
        model = models[max_idx]
    return model


def dataloader_to_nparrays(w_transform, train_loader, test_loader):
    w_transform = w_transform.to('cpu')
    J = w_transform.J
    X = []
    y = []
    for data, labels in train_loader:
        data_t = w_transform(data)
        for j in range(J):
            if j == 0:
                x = deepcopy(data_t[j].detach())
            else:
                x = torch.cat((x, data_t[j].detach()), axis=2)    
        X.append(x)
        y.append(labels)
    X = torch.cat(X).squeeze().numpy()
    y = torch.cat(y).squeeze().numpy()

    X_test = []
    y_test = []
    for data, labels in test_loader:
        data_t = w_transform(data)
        for j in range(J):
            if j == 0:
                x = deepcopy(data_t[j].detach())
            else:
                x = torch.cat((x, data_t[j].detach()), axis=2)          
        X_test.append(x)
        y_test.append(labels)
    X_test = torch.cat(X_test).squeeze().numpy()
    y_test = torch.cat(y_test).squeeze().numpy()   
    
    return (X, y), (X_test, y_test)


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
    (train_loader, test_loader), model = load_dataloader_and_pretrained_model(p)
    
    # prepare model
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)   

    if p.warm_start is None:
        wt = DWT1d(wave=p.wave, mode='zero', J=p.J, init_factor=p.init_factor, noise_factor=p.noise_factor).to(device)
    else:
        wt = warm_start(p, out_dir)        
    
    # train
    params = list(wt.parameters())
    optimizer = torch.optim.Adam(params, lr=p.lr)
    loss_f = get_loss_f(lamlSum=p.lamlSum, lamhSum=p.lamhSum, lamL2norm=p.lamL2norm, lamCMF=p.lamCMF, lamConv=p.lamConv, lamL1wave=p.lamL1wave, lamL1attr=p.lamL1attr)
    trainer = Trainer(model, wt, optimizer, loss_f, target=p.target, 
                      use_residuals=True, attr_methods=p.attr_methods, device=device, n_print=50)       
    # run
    trainer(train_loader, epochs=p.num_epochs)    
    
    # calculate losses
    print('calculating losses and metric...')   
    validator = Validator(model, test_loader)
    rec_loss, lsum_loss, hsum_loss, L2norm_loss, CMF_loss, conv_loss, L1wave_loss, L1saliency_loss, L1inputxgrad_loss = validator(wt, target=p.target)
    s.train_losses = trainer.train_losses
    s.rec_loss = rec_loss
    s.lsum_loss = lsum_loss
    s.hsum_loss = hsum_loss
    s.L2norm_loss = L2norm_loss
    s.CMF_loss = CMF_loss
    s.conv_loss = conv_loss
    s.L1wave_loss = L1wave_loss
    s.L1saliency_loss = L1saliency_loss
    s.L1inputxgrad_loss = L1inputxgrad_loss
    s.net = wt    
    
    # save
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, p._str(p) + '.pkl'), 'wb'))    
    torch.save(wt.state_dict(), opj(out_dir, p._str(p) + '.pth')) 