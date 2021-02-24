import abc
import math
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

sys.path.append('../../')
sys.path.append('../../../')
from adaptive_wavelets.vae.utils.math import matrix_log_density_gaussian, log_density_gaussian, log_importance_weight_matrix, logsumexp
from adaptive_wavelets.vae.models.loss_hessian import hessian_penalty

RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def get_loss_f(**kwargs_parse):
    """Return the loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    return Loss(beta=kwargs_parse["beta"],
                lamPT=kwargs_parse["lamPT"],
                lamNN=kwargs_parse["lamNN"],
                lamH=kwargs_parse["lamH"],
                lamSP=kwargs_parse["lamSP"],
                decoder=kwargs_parse["decoder"],
                **kwargs_all)



class BaseLoss(abc.ABC):
    """
    Base class for losses.
    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.
    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.
    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.
        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).
        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).
        is_train : bool
            Whether currently in train mode.
        storer : dict
            Dictionary in which to store important variables for vizualisation.
        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer
    


class Loss(BaseLoss):
    """
    """
    def __init__(self, beta=0., lamPT=0., lamNN=0., lamH=0., lamSP=0.,
                 is_mss=True, decoder=None, **kwargs):
        """
        Parameters
        ----------
        beta : float
            Hyperparameter for beta-VAE term.
            
        lamPT : float
            Hyperparameter for penalizing change in one latent induced by another.
            
        lamNN : float
            Hyperparameter for penalizing distance to nearest neighbors in each batch
            
        lamH : float
            Hyperparameter for penalizing Hessian
        
        lamSP : float
            Hyperparameter for sparisty of Jacobian
            
        decoder: func
            Torch module which maps from latent space to reconstruction            
        """    
        super().__init__(**kwargs)
        self.beta = beta
        self.lamPT = lamPT
        self.lamNN = lamNN        
        self.lamH = lamH
        self.lamSP = lamSP
        self.is_mss = is_mss
        self.decoder = decoder

    def __call__(self, data, recon_data, latent_dist, is_train, storer, 
                 latent_sample=None, latent_output=None, n_data=None):
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
            
        latent_output: torch.Tensor, optional
            Output of the Decoder->Encoder mapping of latent sample. Shape : (batch_size, latent_dim).
            
        n_data: int, optional
            Total number of training examples.             

        Return
        ------
        loss : torch.Tensor
        """        
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape
        
        self.rec_loss = _reconstruction_loss(data, recon_data,
                                             storer=storer, 
                                             distribution=self.rec_dist)
        self.kl_loss = _kl_normal_loss(*latent_dist, storer)
        
        # total loss
        loss = self.rec_loss + (self.beta * self.kl_loss)                         
        
        # pointwise independence loss
        self.pt_loss = 0
        if self.lamPT > 0 and latent_output is not None:
            jac = jacobian(latent_output, latent_sample)
            for i in range(latent_dim):
                jac[:,i,i] = 0 # make partial i / partial i zero
            self.pt_loss += abs(jac).sum()/batch_size
            loss += self.lamPT * self.pt_loss 
        
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
            self.hessian_loss += hessian_penalty(self.decoder, latent_sample, k=10)
            loss += self.lamH * self.hessian_loss
    
        # sparsity loss
        self.sp_loss = 0
        if self.lamSP > 0:
            decoded_data = self.decoder(latent_sample)
            self.sp_loss += abs(jacobian(decoded_data, latent_sample)).sum()/batch_size
            loss += self.lamSP * self.sp_loss
            
        # total correlation loss
        if n_data is not None:
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
            
        if storer is not None:
            storer['loss'].append(loss.item())
            storer['pt_loss'].append(self.pt_loss.item())
            storer['nn_loss'].append(self.nearest_neighbor_loss.item())
            storer['h_loss'].append(self.hessian_loss.item())
            storer['sp_loss'].append(self.sp_loss.item())      
        
        return loss
            


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
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
    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.
    storer : dict
        Dictionary in which to store important variables for vizualisation.
        
    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size = recon_data.size(0)

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        loss = F.mse_loss(recon_data, data, reduction="sum")
    elif distribution == "laplace":
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
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
    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


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

    
    