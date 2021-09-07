from copy import deepcopy

import torch

from awave.losses import _reconstruction_loss, _lsum_loss, _hsum_loss, _L2norm_loss, _CMF_loss, \
    _conv_loss, _L1_wave_loss, _L1_attribution_loss
from awave.utils.wave_attributions import Attributer
from awave.trim import TrimModel


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
        is_parallel = 'data_parallel' in str(type(w_transform))
        wt_inverse = w_transform.module.inverse if is_parallel else w_transform.inverse  # use multiple GPUs or not
        mt = TrimModel(self.model, wt_inverse, use_residuals=self.use_residuals)

        Saliency = Attributer(mt, attr_methods='Saliency', is_train=False, device=self.device)
        Inputxgrad = Attributer(mt, attr_methods='InputXGradient', is_train=False, device=self.device)

        rec_loss = 0.
        lsum_loss = 0.
        hsum_loss = 0.
        L2norm_loss = 0.
        CMF_loss = 0.
        conv_loss = 0.
        L1wave_loss = 0.
        L1saliency_loss = 0.
        L1inputxgrad_loss = 0.
        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(self.device)
            data_t = w_transform(data)
            recon_data = wt_inverse(data_t)
            saliency = Saliency(data_t, target=target, additional_forward_args=deepcopy(data))
            inputxgrad = Inputxgrad(data_t, target=target, additional_forward_args=deepcopy(data))

            rec_loss += _reconstruction_loss(data, recon_data).item()
            lsum_loss += _lsum_loss(w_transform.module).item() if is_parallel else _lsum_loss(w_transform).item()
            hsum_loss += _hsum_loss(w_transform.module).item() if is_parallel else _hsum_loss(w_transform).item()
            L2norm_loss += _L2norm_loss(w_transform.module).item() if is_parallel else _L2norm_loss(w_transform).item()
            CMF_loss += _CMF_loss(w_transform.module).item() if is_parallel else _CMF_loss(w_transform).item()
            conv_loss += _conv_loss(w_transform.module).item() if is_parallel else _conv_loss(w_transform).item()
            L1wave_loss += _L1_wave_loss(data_t).item()
            L1saliency_loss += _L1_attribution_loss(saliency).item()
            L1inputxgrad_loss += _L1_attribution_loss(inputxgrad).item()

        mean_rec_loss = rec_loss / (batch_idx + 1)
        mean_lsum_loss = lsum_loss / (batch_idx + 1)
        mean_hsum_loss = hsum_loss / (batch_idx + 1)
        mean_L2norm_loss = L2norm_loss / (batch_idx + 1)
        mean_CMF_loss = CMF_loss / (batch_idx + 1)
        mean_conv_loss = conv_loss / (batch_idx + 1)
        mean_L1wave_loss = L1wave_loss / (batch_idx + 1)
        mean_L1saliency_loss = L1saliency_loss / (batch_idx + 1)
        mean_L1inputxgrad_loss = L1inputxgrad_loss / (batch_idx + 1)
        return (mean_rec_loss, mean_lsum_loss, mean_hsum_loss,
                mean_L2norm_loss, mean_CMF_loss, mean_conv_loss,
                mean_L1wave_loss, mean_L1saliency_loss, mean_L1inputxgrad_loss)
