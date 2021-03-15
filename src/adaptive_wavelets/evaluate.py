import numpy as np
import torch
import os, sys
opj = os.path.join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from copy import deepcopy
from wave_attributions import Attributer
from losses import _L1_attribution_loss, _reconstruction_loss, _sum_loss, _L2norm_loss, _CMF_loss, _L1_wave_loss

sys.path.append('../../lib/trim')
from trim import TrimModel

    
class Validator():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: torch.model
    
    data_loader: torch.utils.data.DataLoader
        
    device: torch.device, optional
        Device on which to run the code.
        
    use_residuals : boolean, optional
        Use residuals to compute TRIM score.
    """
    def __init__(self, model, data_loader, 
                 device=torch.device("cuda"),
                 use_residuals=True):

        self.device = device
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.use_residuals = use_residuals
 
    def __call__(self, w_transform, target=1):
        """
        Tests the model for one epoch.

        Parameters
        ----------
        w_transform: torch.nn.module
            Wavelet transformer

        Return
        ------
        mean_epoch_loss: float
        """
        w_transform = w_transform.to(self.device)
        w_transform = w_transform.eval()
        mt = TrimModel(self.model, w_transform.inverse, use_residuals=self.use_residuals) 
        
        Saliency = Attributer(mt, attr_methods='Saliency', is_train=False, device=self.device)
        Inputxgrad = Attributer(mt, attr_methods='InputXGradient', is_train=False, device=self.device)
        
        rec_loss = 0.
        sum_loss = 0.
        L2norm_loss = 0.
        CMF_loss = 0.
        L1wave_loss = 0.        
        L1saliency_loss = 0.
        L1inputxgrad_loss = 0.
        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(self.device)
            data_t = w_transform(data)
            recon_data = w_transform.inverse(data_t)
            saliency = Saliency(data_t, target=target, additional_forward_args=deepcopy(data))
            inputxgrad = Inputxgrad(data_t, target=target, additional_forward_args=deepcopy(data))
            
            rec_loss += _reconstruction_loss(data, recon_data).item()
            sum_loss += _sum_loss(w_transform).item()
            L2norm_loss += _L2norm_loss(w_transform).item()
            CMF_loss += _CMF_loss(w_transform).item()
            L1wave_loss += _L1_wave_loss(data_t).item() 
            L1saliency_loss += _L1_attribution_loss(saliency).item()
            L1inputxgrad_loss += _L1_attribution_loss(inputxgrad).item()

        mean_rec_loss = rec_loss / (batch_idx + 1)
        mean_sum_loss = sum_loss / (batch_idx + 1)
        mean_L2norm_loss = L2norm_loss / (batch_idx + 1)
        mean_CMF_loss = CMF_loss / (batch_idx + 1)
        mean_L1wave_loss = L1wave_loss / (batch_idx + 1)
        mean_L1saliency_loss = L1saliency_loss / (batch_idx + 1)
        mean_L1inputxgrad_loss = L1inputxgrad_loss / (batch_idx + 1)
        return (mean_rec_loss, mean_sum_loss, mean_L2norm_loss, mean_CMF_loss, mean_L1wave_loss, mean_L1saliency_loss, mean_L1inputxgrad_loss)
    
    
        