import numpy as np
import torch
import random
import os,sys
opj = os.path.join
from copy import deepcopy
import pickle as pkl
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# adaptive-wavelets modules
sys.path.append('../../src/adaptive_wavelets')
from losses import get_loss_f
from train import Trainer
from evaluate import Validator
from transform1d import DWT1d
from wave_attributions import Attributer

sys.path.append('../../src/models')
sys.path.append('../../src/dsets/mnist')
from dset import get_dataloader, load_pretrained_model

sys.path.append('../../src')
from warmstart import warm_start


class p:
    """Parameters for simulated data
    """
    # data & model path
    data_path = "../../src/dsets/mnist/data"
    model_path = "../../src/dsets/mnist/data"
    wt_type = 'DWT1d'
    
    # parameters for wavelet initialization
    wave = 'db5'
    J = 4
    init_factor = 1
    noise_factor = 0.3
    const_factor = 0
    
    # parameters for training
    batch_size = 100
    lr = 0.001
    num_epochs = 50
    attr_methods = 'Saliency'
    lamlSum = 1
    lamhSum = 1
    lamL2norm = 1 
    lamCMF = 1
    lamConv = 1
    lamL1wave = 0.1
    lamL1attr = 1     
    target = 0
    
    # run with warmstart
    warm_start = None    
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/adaptive-wavelets/notebooks/simulation/results" 
    dirname = "dirname"
    pid = ''.join(["%s" % random.randint(0, 9) for num in range(0, 10)])

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
    
    
if __name__ == '__main__':    
    args = parser.parse_args()
    for arg in vars(args):
        setattr(p, arg, getattr(args, arg))
    
    # create dir
    out_dir = opj(p.out_dir, p.dirname)
    os.makedirs(out_dir, exist_ok=True)
    
    # load data and model
    train_loader, test_loader = get_dataloader(p.data_path,
                                               batch_size=p.batch_size)

    model = load_pretrained_model(p.model_path)    
    
    # prepare model
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)   

    if p.warm_start is None:
        wt = DWT1d(wave=p.wave, mode='zero', J=p.J, 
                   init_factor=p.init_factor, 
                   noise_factor=p.noise_factor,
                   const_factor=p.const_factor).to(device)
        wt.train()
    else:
        wt = warm_start(p, out_dir).to(device)        
        wt.train()
        
    # check if we have multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)        
        wt = torch.nn.DataParallel(wt)
    
    # train
    params = list(wt.parameters())
    optimizer = torch.optim.Adam(params, lr=p.lr)
    loss_f = get_loss_f(lamlSum=p.lamlSum, lamhSum=p.lamhSum, lamL2norm=p.lamL2norm, lamCMF=p.lamCMF, lamConv=p.lamConv, lamL1wave=p.lamL1wave, lamL1attr=p.lamL1attr)
    trainer = Trainer(model, wt, optimizer, loss_f, target=p.target, 
                      use_residuals=True, attr_methods=p.attr_methods, device=device, n_print=5)      
    
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
    if torch.cuda.device_count() > 1:
        torch.save(wt.module.state_dict(), opj(out_dir, p._str(p) + '.pth'))   
    else:
        torch.save(wt.state_dict(), opj(out_dir, p._str(p) + '.pth'))     
    
  