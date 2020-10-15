import numpy as np
import os, sys
opj = os.path.join
import logging
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F
from copy import deepcopy

# trim modules
sys.path.append('../../lib/trim')
sys.path.append('../../../lib/trim')
from trim import DecoderEncoder


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: torch.model
    
    optimizer: torch.optim.Optimizer
    
    loss_f: vae.models.BaseLoss
        Loss function.
        
    device: torch.device, optional
        Device on which to run the code.
        
    use_residuals : boolean, optional
        Use residuals to map latent to latent.
    """
    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 use_residuals=True):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.use_residuals = use_residuals
        self._create_latent_map()   
    

    def __call__(self, train_loader, test_loader=None, epochs=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.
        """
        self.train_losses = np.empty(epochs)
        self.test_losses = np.empty(epochs)
        for epoch in range(epochs):
            if test_loader is not None:
                mean_epoch_loss = self._train_epoch(train_loader)
                mean_epoch_test_loss = self._test_epoch(test_loader)
                print('====> Epoch: {} Average train loss: {:.4f} (Test set loss: {:.4f})'.format(epoch, mean_epoch_loss, 
                                                                                                  mean_epoch_test_loss))
                self.train_losses[epoch] = mean_epoch_loss
                self.test_losses[epoch] = mean_epoch_test_loss
                
            else:
                mean_epoch_loss = self._train_epoch(train_loader)
                print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, mean_epoch_loss))
                self.train_losses[epoch] = mean_epoch_loss    

        
    def _create_latent_map(self):
        """
        Create saliency object for decoder-encoder map.
        
        Parameters
        ----------
        """
        self.latent_map = DecoderEncoder(self.model, use_residuals=self.use_residuals)

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        self.model.train()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            iter_loss = self._train_iteration(data, n_data=None)
            epoch_loss += iter_loss         

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        self.model.eval()
        return mean_epoch_loss

    def _train_iteration(self, data, n_data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
            
        """
        data = data.to(self.device)

        recon_data, latent_dist, latent_sample = self.model(data)
        latent_output = self.latent_map(latent_sample, data)
        loss = self.loss_f(data, recon_data, latent_dist, self.model.training, storer=None,
                           latent_sample=latent_sample, latent_output=latent_output, n_data=None)  
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss.item()  
    
    def _test_epoch(self, data_loader):
        """
        Tests the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        self.model.eval()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)
            recon_data, latent_dist, latent_sample = self.model(data)
            latent_output = self.latent_map(latent_sample, data)
            loss = self.loss_f(data, recon_data, latent_dist, self.model.training, storer=None,
                               latent_sample=latent_sample, latent_output=latent_output, n_data=None)                              
            iter_loss = loss.item()
            epoch_loss += iter_loss       

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        return mean_epoch_loss    
    
    
    