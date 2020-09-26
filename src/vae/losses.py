import abc
import math
import os, sys
sys.path.append('../../')
sys.path.append('../../../')
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from src.vae.utils import matrix_log_density_gaussian, log_density_gaussian, log_importance_weight_matrix, logsumexp
from src.vae.loss_hessian import hessian_penalty


class Loss(abc.ABC):
    """
    """
    def __init__(self, beta=0., mu=0., lamPT=0., lamCI=0., lamNN=0., lamH=0., lamSP=0.,
                 alpha=0., gamma=0., tc=0., is_mss=True, decoder=None):
        """
        Parameters
        ----------
        beta : float
            Hyperparameter for beta-VAE term.

        mu : float
            Hyperparameter for latent distribution mean.
            
        lamPT : float
            Hyperparameter for penalizing change in one latent induced by another.
            
        lamCI : float
            Hyperparameter for penalizing change in conditional distribution p(z_-j | z_j).
            
        lamNN : float
            Hyperparameter for penalizing distance to nearest neighbors in each batch
            
        lamH : float
            Hyperparameter for penalizing Hessian
        
        lamSP : float
            Hyperparameter for sparisty of Jacobian
            
        alpha : float
            Hyperparameter for mutual information term.
            
        gamma: float
            Hyperparameter for dimension-wise KL term.
            
        tc: float
            Hyperparameter for total correlation term.
            
        decoder: func
            Torch module which maps from latent space to reconstruction            
        """           
        self.beta = beta
        self.mu = mu
        self.lamPT = lamPT
        self.lamCI = lamCI
        self.lamNN = lamNN        
        self.lamH = lamH
        self.lamSP = lamSP
        self.alpha = alpha
        self.gamma = gamma
        self.tc = tc
        self.is_mss = is_mss
        self.decoder = decoder

    def __call__(self, data, recon_data, latent_dist, latent_sample, n_data,
                 latent_output=None):
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
            
        latent_dist: list of torch.Tensor
            Encoder latent distribution [mean, logvar]. Shape : (batch_size, latent_dim).
            
        latent_sample: torch.Tensor
            Latent samples. Shape : (batch_size, latent_dim).
            
        n_data: int
            Total number of training examples. 
            
        latent_output: torch.Tensor, optional
            Output of the Decoder->Encoder mapping of latent sample. Shape : (batch_size, latent_dim).

        Return
        ------
        loss : torch.Tensor
        """        
        batch_size, latent_dim = latent_sample.shape
        
        self.rec_loss = _reconstruction_loss(data, recon_data)
        self.kl_loss = _kl_normal_loss(*latent_dist) 
        self.mu_loss = _kl_normal_loss(latent_dist[0], torch.zeros_like(latent_dist[1])) 

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
                                self.mu * self.mu_loss)        
        
        # pointwise independence loss
        self.pt_loss = 0
        if self.lamPT > 0 and latent_output is not None:
            jac = jacobian(latent_output, latent_sample)
            for i in range(latent_dim):
                jac[:,i,i] = 0 # make partial i / partial i zero
            self.pt_loss += abs(jac).sum()/batch_size
            loss += self.lamPT * self.pt_loss
        
        # local independence loss
        self.ci_loss = 0
        if self.lamCI > 0:
            log_q_zCzi = log_qz.view(batch_size, 1) - log_qzi
            jac = jacobian(log_q_zCzi, latent_sample)
            for i in range(latent_dim):
                self.ci_loss += abs(jac[:,i,i]).mean()     
            loss += self.lamCI * self.ci_loss        
        
        # nearest-neighbor batch loss
        self.nearest_neighbor_loss = 0
        if self.lamNN > 0:
            for i in range(batch_size):
                dists = torch.pairwise_distance(latent_sample[i], latent_sample)
                self.nearest_neighbor_loss += dists.sort()[0][1] # exclude distance to itself
            loss += self.lamNN * self.nearest_neighbor_loss
            
        # Hessian loss
        self.hessian_loss = 0
        if self.lamH > 0:
            # print('calculating hessian loss...')
            self.hessian_loss += hessian_penalty(self.decoder, latent_sample)
            loss += self.lamH * self.hessian_loss
    
        # sparsity loss
        self.sp_loss = 0
        if self.lamSP > 0:
            decoded_data = self.decoder(latent_sample)
            self.sp_loss += abs(jacobian(decoded_data, latent_sample)).sum()/batch_size
            loss += self.lamSP * self.sp_loss
        
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


def _get_log_qz_qzi_perb(latent_sample_perb, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim, perb_size = latent_sample_perb.shape
    mu, logvar = latent_dist
    
    latent_sample_perb = latent_sample_perb.view(batch_size, 1, hidden_dim, perb_size)    
    mu = mu.view(1, batch_size, hidden_dim, 1)
    logvar = logvar.view(1, batch_size, hidden_dim, 1)
    
    # calculate log q(z)
    mat_log_qz = log_density_gaussian(latent_sample_perb, mu, logvar)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_perb.device)
        mat_log_qzi = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1, 1)
        
    log_qz = logsumexp(mat_log_qz.sum(2) + log_iw_mat.view(batch_size, batch_size, 1), dim=1, keepdim=False)
    log_qzi = logsumexp(mat_log_qzi, dim=1, keepdim=False)

    return log_qz, log_qzi


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]
    
    y : torch.Tensor Size: (batch_size, y_dim)
    x : torch.Tensor Size: (batch_size, x_dim)
    
    Return
    ------
    jac : torch.Tensor Size: (batch_size, y_dim, x_dim) 
        Jacobian of y w.r.t. x
    """
    grads = []
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:,i] = 1
        grads.append(gradient(y, x, grad_outputs=grad_outputs).unsqueeze(1))
    jac = torch.cat(grads, dim=1)
    return jac
            
    
    