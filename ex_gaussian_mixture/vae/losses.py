import abc
import math
import os,sys
sys.path.append('../disentangling-vae')

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from utils import *


class Loss(abc.ABC):
    """
    """
    def __init__(self, beta=4., attr=1., alpha=0., gamma=0., tc=1., is_mss=True):
        self.beta = beta
        self.attr = attr
        self.alpha = alpha
        self.gamma = gamma
        self.tc = tc
        self.is_mss = is_mss
        self.L1Loss = torch.nn.L1Loss()

    def __call__(self, data, recon_data, latent_dist, latent_sample, n_data):
        batch_size, latent_dim = latent_sample.shape
        
        self.rec_loss = _reconstruction_loss(data, recon_data)
        self.kl_loss = _kl_normal_loss(*latent_dist)

        log_pz, log_qz, log_qzi, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                                      latent_dist,
                                                                                      n_data,
                                                                                      is_mss=self.is_mss)      
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        self.mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        self.tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        self.dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # total loss
        loss = self.rec_loss + (self.beta * self.kl_loss +
                                self.alpha * self.mi_loss +
                                self.tc * self.tc_loss +
                                self.gamma * self.dw_kl_loss)        
        
        if self.attr > 0:
            self.attr_loss = 0
            log_q_zCzi = log_qz.view(batch_size, 1) - log_qzi
            # local indepedence loss
            eps = 0.1
            for i in range(latent_dim):
                perb = torch.zeros_like(latent_sample)
                perb[:,i] = eps * torch.ones(batch_size)
                latent_sample_p = latent_sample + perb
                
                _, log_qz_p, log_qzi_p, _, _ = _get_log_pz_qz_prodzi_qzCx(latent_sample_p,
                                                                          latent_dist,
                                                                          n_data,
                                                                          is_mss=self.is_mss)                 
                log_q_zCzi_p = log_qz_p.view(batch_size, 1) - log_qzi_p    
                diff = (log_q_zCzi - log_q_zCzi_p)[:,i]
                self.attr_loss += self.L1Loss(diff, torch.zeros_like(diff))  
            loss += self.attr * self.attr_loss

        return loss
    
    
def _reconstruction_loss(data, recon_data):
    """
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
    """
    batch_size, dim = recon_data.size()

    loss = F.mse_loss(recon_data, data, reduction="sum") 
    loss = loss / batch_size

    return loss


def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl   


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
    
    # calculate log q(z)
    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qzi = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
        
    log_qz = logsumexp(mat_log_qz.sum(2) + log_iw_mat, dim=1, keepdim=False)
    log_qzi = logsumexp(mat_log_qzi, dim=1, keepdim=False)
    log_prod_qzi = log_qzi.sum(1)

    return log_pz, log_qz, log_qzi, log_prod_qzi, log_q_zCx


#         if self.attr > 0:
#             log_q_zCzi = log_qz.view(batch_size, 1) - log_qzi
            
#             # local indepedence loss
#             for i in range(latent_dim):
#                 gradients = torch.autograd.grad(log_q_zCzi[:,i], latent_sample, grad_outputs=torch.ones_like(log_q_zCzi[:,i]), 
#                                                 retain_graph=True, create_graph=True, only_inputs=True)[0][:,i]               
#                 loss += self.attr * self.L1Loss(gradients, torch.zeros_like(gradients))  