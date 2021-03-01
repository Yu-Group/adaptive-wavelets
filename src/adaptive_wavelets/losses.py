import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_f(**kwargs_parse):
    """Return the loss function given the argparse arguments."""
    return Loss(lamL1attr=kwargs_parse["lamL1attr"])


class Loss():
    """
    """
    def __init__(self, lamL1attr=0.):
        """
        Parameters
        ----------
        lamL1attr : float
            Hyperparameter for penalizing L1 norm of attributions
        """    
        self.lamL1attr = lamL1attr

    def __call__(self, data, recon_data, data_t, attributions=None):
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
            
        data_t: list of torch.Tensor
            Input data after wavelet transform.
            
        attributions: torch.Tensor
            Input attribution scores.          

        Return
        ------
        loss : torch.Tensor
        """        
        self.rec_loss = _reconstruction_loss(data, recon_data)
        
        # L1 penalty on attributions
        self.L1attr_loss = 0
        if self.lamL1attr > 0:
            self.L1attr_loss += _L1_attribution_loss(attributions)

        # total loss
        loss = self.rec_loss + self.lamL1attr*self.L1attr_loss 
        
        return loss
            


def _reconstruction_loss(data, recon_data):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.
    
    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).
    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).
        
    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size = recon_data.size(0)
    loss = F.mse_loss(recon_data, data, reduction="sum")
    loss = loss / batch_size

    return loss


def _L1_attribution_loss(attributions):
    """
    Calculate L1 norm of the attributions
    """
    batch_size = attributions[0].size(0)
    loss = tuple_L1Loss(attributions)
    loss = loss / batch_size

    return loss


def tuple_Sum(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(x[i])
    return output/num    


def tuple_L1Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(abs(x[i]))
    return output/num


def tuple_L2Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(x[i]**2)
    return output/num  