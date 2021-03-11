import numpy as np
import torch
import os, sys
opj = os.path.join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from copy import deepcopy

sys.path.append('../../lib/trim')
from trim import TrimModel


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: torch.model
    
    optimizer: torch.optim.Optimizer
    
    w_transform: torch.nn.module
        Wavelet transformer
        
    attributer: torch.nn.module
        Attributer for model appended with wavelet transform
        
    device: torch.device, optional
        Device on which to run the code.
        
    use_residuals : boolean, optional
        Use residuals to compute TRIM score.
    """
    def __init__(self, model, w_transform, attributer, optimizer, loss_f,
                 target=1,
                 device=torch.device("cuda"),
                 use_residuals=True,
                 attr_methods='InputXGradient',
                 n_print=1):

        self.device = device
        self.model = model.to(self.device)
        self.w_transform = w_transform.to(self.device)
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.mt = TrimModel(model, w_transform.inverse, use_residuals=use_residuals)    
        self.attributer = attributer(self.mt, attr_methods=attr_methods, device=self.device)
        self.target = target
        self.n_print = n_print

    def __call__(self, train_loader, test_loader=None, epochs=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.
        """
        print("Starting Training Loop...")
        self.train_losses = np.empty(epochs)
        self.test_losses = np.empty(epochs)
        for epoch in range(epochs):
            if test_loader is not None:
                mean_epoch_loss = self._train_epoch(train_loader, epoch)
                mean_epoch_test_loss = self._test_epoch(test_loader)
                if epoch % self.n_print == 0:
                    print('\n====> Epoch: {} Average train loss: {:.4f} (Test set loss: {:.4f})'.format(epoch, mean_epoch_loss, 
                                                                                                      mean_epoch_test_loss))
                self.train_losses[epoch] = mean_epoch_loss
                self.test_losses[epoch] = mean_epoch_test_loss
                
            else:
                mean_epoch_loss = self._train_epoch(train_loader, epoch)
                if epoch % self.n_print == 0:
                    print('\n====> Epoch: {} Average train loss: {:.4f}'.format(epoch, mean_epoch_loss))
                self.train_losses[epoch] = mean_epoch_loss    

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
        self.w_transform.train()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss   
            if epoch % self.n_print == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), iter_loss), end='')              

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        self.w_transform.eval()
        return mean_epoch_loss

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
            
        """
        data = data.to(self.device)
        # zero grad
        self.optimizer.zero_grad()
        # transform
        data_t = self.w_transform(data)
        # reconstruction
        recon_data = self.w_transform.inverse(data_t)
        # TRIM score
        with torch.backends.cudnn.flags(enabled=False):
            attributions = self.attributer(data_t, target=self.target, additional_forward_args=deepcopy(data)) if self.loss_f.lamL1attr > 0 else None
        # loss
        loss = self.loss_f(self.w_transform, data, recon_data, data_t, attributions)

        # backward
        loss.backward()
        # update step
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
        self.w_transform.eval()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)
            data_t = self.w_transform(data)
            recon_data = self.w_transform.inverse(data_t)
            attributions = self.attributer(data_t, target=self.target, additional_forward_args=deepcopy(data))
            loss = self.loss_f(self.w_transform, data, recon_data, data_t, attributions)                  
            iter_loss = loss.item()
            epoch_loss += iter_loss   
            print('\rTest: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), iter_loss), end='')               

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        return mean_epoch_loss 
    
    
class Validator():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: torch.model
    
    optimizer: torch.optim.Optimizer
    
    w_transform: torch.nn.module
        Wavelet transformer
        
    attributer: torch.nn.module
        Attributer for model appended with wavelet transform
        
    device: torch.device, optional
        Device on which to run the code.
        
    use_residuals : boolean, optional
        Use residuals to compute TRIM score.
    """
    def __init__(self, model, w_transform, attributer, loss_f,
                 target=1,
                 device=torch.device("cpu"),
                 attr_methods='InputXGradient',
                 use_residuals=True):

        self.device = device
        self.model = model.to(self.device)
        self.w_transform = w_transform.to(self.device)
        self.loss_f = loss_f
        self.mt = TrimModel(model, w_transform.inverse, use_residuals=use_residuals)    
        self.attributer = attributer(self.mt, attr_methods=attr_methods, device=self.device)
        self.target = target
 
    def __call__(self, data_loader):
        """
        Tests the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        Return
        ------
        mean_epoch_loss: float
        """
        self.w_transform.eval()
        epoch_loss = 0.
        rec_loss = 0.
        sum_loss = 0.
        L2norm_loss = 0.
        CMF_loss = 0.
        L1wave_loss = 0.        
        L1attr_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)
            data_t = self.w_transform(data)
            recon_data = self.w_transform.inverse(data_t)
            attributions = self.attributer(data_t, target=self.target, additional_forward_args=deepcopy(data))
            loss = self.loss_f(self.w_transform, data, recon_data, data_t, attributions)                  
            iter_loss = loss.item()
            epoch_loss += iter_loss   
            print('\rTest: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), iter_loss), end='')   
            rec_loss += self.loss_f.rec_loss.item()
            sum_loss += self.loss_f.sum_loss.item()
            L2norm_loss += self.loss_f.L2norm_loss.item()
            CMF_loss += self.loss_f.CMF_loss.item()
            L1wave_loss += self.loss_f.L1wave_loss.item()
            L1attr_loss += self.loss_f.L1attr_loss.item()

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        mean_rec_loss = rec_loss / (batch_idx + 1)
        mean_sum_loss = sum_loss / (batch_idx + 1)
        mean_L2norm_loss = L2norm_loss / (batch_idx + 1)
        mean_CMF_loss = CMF_loss / (batch_idx + 1)
        mean_L1wave_loss = L1wave_loss / (batch_idx + 1)
        mean_L1attr_loss = L1attr_loss / (batch_idx + 1)
        return (mean_epoch_loss, mean_rec_loss, mean_sum_loss, mean_L2norm_loss, mean_CMF_loss, mean_L1wave_loss, mean_L1attr_loss)
    
    
        