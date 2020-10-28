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

from models import AutoEncoder, AutoEncoderSimple, load_model

sys.path.append('../../src')
sys.path.append('../../src/vae')
sys.path.append('../../src/vae/models')
sys.path.append('../../src/dsets/cosmology')
from dset import get_dataloader
from model import init_specific_model
from losses import get_loss_f, _reconstruction_loss
from training import Trainer

sys.path.append('../../lib/trim')
# trim modules
from trim import DecoderEncoder


parser = argparse.ArgumentParser(description='Cosmology Example')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--h_channels', type=int, default=5,
                   help='number of hidden channels in autoencoder (default: 5)')
parser.add_argument('--dirname', default='vary',
                   help='name of directory')


class p:
    '''Parameters for cosmology data
    '''
    # parameters for generating data
    seed = 1
    data_path = "../../src/dsets/cosmology/data"
    
    # parameters for model architecture
    img_size = (1, 64, 64)
    h_channels = 5
    
    # parameters for training
    train_batch_size = 64
    test_batch_size = 100
    lr = 1*1e-4
    rec_dist = "gaussian"
    reg_anneal = 0
    num_epochs = 50
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/local-vae/notebooks/ex_cosmology/results" # wooseok's setup
#     out_dir = '/scratch/users/vision/chandan/local-vae' # chandan's setup
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 10)])

    def _str(self):
        vals = vars(p)
        return 'h_channels=' + str(vals['h_channels']) + '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
class s:
    '''Parameters to save
    '''
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
def train(train_loader, model, optimizer, device=device): 
    model.train()
    # Training Loop
    # Lists to keep track of progress
    losses = []
    
    print("Starting Training Loop...")
    
    # For each epoch
    for epoch in range(p.num_epochs):
        epoch_loss = 0
        # For each batch in the dataloader
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            recon_data, latent_sample = model(data)
            # loss
            loss = _reconstruction_loss(data, recon_data, distribution="gaussian", storer=None)
            # zero grad
            optimizer.zero_grad()
            # backward
            loss.backward()
            # Update step
            optimizer.step()

            epoch_loss += loss.data.item()
            
        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, mean_epoch_loss))
        # Save Losses for plotting later
        losses.append(mean_epoch_loss)        
    
    model.eval()
    return losses    


    
if __name__ == '__main__':
    args = parser.parse_args()
    for arg in vars(args):
        setattr(p, arg, getattr(args, arg))
    
    # create dir
    out_dir = opj(p.out_dir, p.dirname)
    os.makedirs(out_dir, exist_ok=True)        

    # seed
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)
    
    # get dataloaders
    train_loader = get_dataloader(p.data_path, 
                                  batch_size=p.train_batch_size)    
    
    # prepare model
    model = AutoEncoderSimple(img_size=p.img_size, hid_channels=p.hid_channels).to(device)    

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    losses = trainer(train_loader, model, optimizer, device=device)
    
    # calculate losses
    print('calculating losses and metric...')    
    s.losses = losses
    s.net = model    
    
    # save
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, p._str(p) + '.pkl'), 'wb'))    
    torch.save(model.state_dict(), opj(out_dir, p._str(p) + '.pth'))     