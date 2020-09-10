import numpy as np
import os, sys

from tqdm import trange
import torch
from torch.nn import functional as F
from copy import deepcopy
from utils import *

# trim modules
sys.path.append('../trim')
from trim import TrimModel, DecoderEncoder

class Trainer():
    """
    Class to handle training of model.
    """
    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 use_residuals=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.use_residuals = use_residuals
        self._create_latent_map()        

    def __call__(self, train_loader, test_loader, epochs=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(train_loader, epoch)
            mean_epoch_test_loss = self._test_epoch(test_loader)
            print('====> Epoch: {} Average train loss: {:.4f} (Test set loss: {:.4f})'.format(epoch, mean_epoch_loss, 
                                                                                              mean_epoch_test_loss))
        
    def _create_latent_map(self):
        """
        Create saliency object for decoder-encoder map.
        
        Parameters
        ----------
        """
        self.latent_map = DecoderEncoder(self.model, use_residuals=self.use_residuals)

    def _train_epoch(self, data_loader, epoch):
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
        n_data = data_loader.dataset.data.shape[0]
        epoch_loss = 0.
        for batch_idx, data in enumerate(data_loader):
            iter_loss = self._train_iteration(data, n_data)
            epoch_loss += iter_loss
            
            if batch_idx % 10 == -1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader),
                    epoch_loss / (batch_idx+1)))            

        mean_epoch_loss = epoch_loss / len(data_loader)
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
        loss = self.loss_f(data, recon_data, latent_dist, latent_sample, n_data, latent_output)  
        
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
        n_data = data_loader.dataset.data.shape[0]
        epoch_loss = 0.
        for batch_idx, data in enumerate(data_loader):
            data = data.to(self.device)
            recon_data, latent_dist, latent_sample = self.model(data)
            latent_output = self.latent_map(latent_sample, data)
            loss = self.loss_f(data, recon_data, latent_dist, latent_sample, n_data, latent_output)                  
            iter_loss = loss.item()
            epoch_loss += iter_loss       

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss    