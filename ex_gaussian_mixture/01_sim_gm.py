%load_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
import acd
from random import randint
from copy import deepcopy
from model import init_specific_model
from losses import Loss
from dset import *
from training import Trainer
from utils import *
from sim_gaussian_mixture import *

if __name__ == '__main__':
    
    # parameters
    attrs = np.linspace(0, 5, 30)
    betas = [0]

    # model hidden dim 
    hidden_dims = [8]

    # seed number
    seeds = [13]

    for attr in attrs:
        p.attr = attr
        p.beta = betas[0]
        p.hidden_dim = hidden_dims[0]
        p.seed = seeds[0]
 
        # seed
        random.seed(p.seed)
        np.random.seed(p.seed)
        torch.manual_seed(p.seed)

        # GET DATALOADERS
        (train_loader, train_latents), (test_loader, test_latents) = define_dataloaders(p)

        # PREPARES MODEL
        model = init_specific_model(orig_dim=p.orig_dim, latent_dim=p.latent_dim, hidden_dim=p.hidden_dim)
        model = model.to(device)

        # TRAINS
        optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
        beta = p.beta
        attr = p.attr
        alpha = p.alpha
        gamma = p.gamma
        tc = p.tc
        num_epochs = p.num_epochs

        loss_f = Loss(beta=beta, attr=attr, alpha=alpha, gamma=gamma, tc=tc, is_mss=True)
        trainer = Trainer(model, optimizer, loss_f, device=device)
        trainer(train_loader, test_loader, epochs=num_epochs)

        # calculate losses
        print('calculating losses and metric...')    
        rec_loss, kl_loss, mi_loss, tc_loss, dw_kl_loss, attr_loss = calc_losses(model, test_loader, loss_f)
        s.reconstruction_loss = rec_loss
        s.kl_normal_loss = kl_loss
        s.total_correlation = tc_loss
        s.mutual_information = mi_loss
        s.dimensionwise_kl_loss = dw_kl_loss
        s.attribution_loss = attr_loss
        s.disentanglement_metric = calc_disentangle_metric(model, test_loader).mean()
        s.net = model    

        # save
        os.makedirs(p.out_dir, exist_ok=True)
        results = {**p._dict(p), **s._dict(s)}
        pkl.dump(results, open(opj(p.out_dir, p._str(p) + '.pkl'), 'wb'))   