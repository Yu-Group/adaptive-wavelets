import os,sys
opj = os.path.join
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
import random

from awd.adaptive_wavelets import DWT1d
from awd.models.models import Feedforward
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader(root_dir, shuffle=True, pin_memory=True, batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.   

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    
    # training data
    X_train, y_train = pkl.load(open(opj(root_dir, 'train.pkl'), 'rb'))
    
    # test data
    X_test, y_test = pkl.load(open(opj(root_dir, 'test.pkl'), 'rb'))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory) 
    test_loader = DataLoader(TensorDataset(X_test, y_test), 
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=pin_memory) 
    
    return train_loader, test_loader


def load_pretrained_model(root_dir, device=device):
    """load pretrained model for interpretation
    """        
    model = Feedforward(input_size=64).to(device)
    model.load_state_dict(torch.load(opj(root_dir, 'FFN.pth')))
    model = model.eval()
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False  
    return model


# PyTorch 
def generate_data(p):
    """simulate 1d data and labels
    """    
    # seed
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    # wavelet transform 
    wt = DWT1d(wave=p.wave_gt, mode='zero', J=p.J, init_factor=1, noise_factor=0)    
    # freeze filter
    for param in wt.parameters():
        param.requires_grad = False  

    X = torch.randn(p.n, 1, p.d)
    X_t = wt(X)

    # true coeff
    beta = ()
    for i in range(len(X_t)):
        beta += (torch.zeros_like(X_t[i][0:1,...]),)
    beta[p.scale_knockout][..., p.idx_knockout - p.window: p.idx_knockout + p.window + 1] = 2.0
    
    # define y
    y = 0
    for x, b in zip(X_t, beta):
        y += torch.matmul(x.squeeze(), b.squeeze())
    eps = p.noise_level * torch.randn_like(y)
    y = y + eps
    
    # training data
    X_train = X[:p.n_train]
    y_train = y[:p.n_train]
               
    # test data
    X_test = X[p.n_train:]
    y_test = y[p.n_train:]      

    return (X_train, y_train), (X_test, y_test)