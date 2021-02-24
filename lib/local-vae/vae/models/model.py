"""
Module containing the main VAE class.
"""
import numpy as np
import torch
from torch import nn
from adaptive_wavelets.vae.models.encoders import get_encoder
from adaptive_wavelets.vae.models.decoders import get_decoder
# from .decoders import get_decoder

MODELS = ["Burgess", "Lin"]


def init_specific_model(model_type, img_size, latent_dim, hidden_dim=None):
    """
    Return an instance of a VAE with encoder and decoder from `model_type`.
    Parameters
    ----------
    img_size : tuple of ints for model_type=Burgess, int for model_type=Lin
        Size or Dimension of images 
    """
    model_type = model_type.lower().capitalize()
    get_enc = get_encoder(model_type)
    get_dec = get_decoder(model_type)
    if model_type == "Burgess":
        encoder = get_enc(img_size, latent_dim)
        decoder = get_dec(img_size, latent_dim)
    elif model_type == "Lin":
        encoder = get_enc(img_size, latent_dim, hidden_dim)
        decoder = get_dec(img_size, latent_dim, hidden_dim)
    else:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))
    
    model = VAE(encoder, decoder)
    model.model_type = model_type  # store to help reloading
    return model



class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        encoder : torch.nn.Module
            class of encoder
        
        decoder : torch.nn.Module
            class of decoder
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
    
    
