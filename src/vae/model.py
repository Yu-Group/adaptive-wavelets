"""
Module containing the encoders.
"""
import numpy as np
import torch
from torch import nn

def init_specific_model(orig_dim, latent_dim, hidden_dim=6):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    encoder = Encoder(orig_dim, latent_dim, hidden_dim)
    decoder = Decoder(orig_dim, latent_dim, hidden_dim)
    model = VAE(encoder, decoder)
    return model


class Encoder(nn.Module):
    def __init__(self, orig_dim=10, latent_dim=2, hidden_dim=6):
        r"""Encoder of the model for GMM samples
        """
        super(Encoder, self).__init__()
        # Layer parameters
        self.orig_dim = orig_dim
        self.latent_dim = latent_dim
        
        # Fully connected layers
        self.lin1 = nn.Linear(self.orig_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.lin3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)
     
    def forward(self, x):
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar    
    
        
class Decoder(nn.Module):
    def __init__(self, orig_dim=10, latent_dim=2, hidden_dim=6):
        r"""Decoder of the model for GMM samples
        """
        super(Decoder, self).__init__()
        # Layer parameters
        self.orig_dim = orig_dim
        self.latent_dim = latent_dim
        
        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.lin3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, orig_dim)

    def forward(self, z):
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = self.lin4(x)

        return x     
    
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Class which defines model and forward pass.
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