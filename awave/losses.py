import numpy as np
import torch
import torch.nn.functional as F

from awave.utils.misc import low_to_high


def get_loss_f(**kwargs_parse):
    """Return the loss function given the argparse arguments."""
    return Loss(lamlSum=kwargs_parse["lamlSum"],
                lamhSum=kwargs_parse["lamhSum"],
                lamL2norm=kwargs_parse["lamL2norm"],
                lamCMF=kwargs_parse["lamCMF"],
                lamConv=kwargs_parse["lamConv"],
                lamL1wave=kwargs_parse["lamL1wave"],
                lamL1attr=kwargs_parse["lamL1attr"])


class Loss():
    """Class of calculating loss functions
    """

    def __init__(self, lamlSum=1., lamhSum=1., lamL2norm=1., lamCMF=1., lamConv=1., lamL1wave=1., lamL1attr=1.,
                 lamHighfreq=0.0):
        """
        Parameters
        ----------
        lamlSum : float
            Hyperparameter for penalizing sum of lowpass filter
            
        lamhSum : float
            Hyperparameter for penalizing sum of highpass filter            
            
        lamL2norm : float
            Hyperparameter to enforce unit norm of lowpass filter
            
        lamCMF : float 
            Hyperparameter to enforce conjugate mirror filter   
            
        lamConv : float
            Hyperparameter to enforce convolution constraint
            
        lamL1wave : float
            Hyperparameter for penalizing L1 norm of wavelet coeffs
        
        lamL1attr : float
            Hyperparameter for penalizing L1 norm of attributions
        """
        self.lamlSum = lamlSum
        self.lamhSum = lamhSum
        self.lamL2norm = lamL2norm
        self.lamCMF = lamCMF
        self.lamConv = lamConv
        self.lamL1wave = lamL1wave
        self.lamL1attr = lamL1attr
        self.lamHighfreq = lamHighfreq

    def __call__(self, w_transform, data, recon_data, data_t, attributions=None):
        """
        Parameters
        ----------
        w_transform : wavelet object
        
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
            
        data_t: list of torch.Tensor
            Input data after wavelet transform.
            
        attributions: torch.Tensor
            Input attribution scores.          

        Return
        ------
        loss : torch.Tensor
        """
        self.rec_loss = _reconstruction_loss(data, recon_data)

        # sum of lowpass filter
        self.lsum_loss = 0
        if self.lamlSum > 0:
            self.lsum_loss += _lsum_loss(w_transform)

        # sum of highpass filter
        self.hsum_loss = 0
        if self.lamhSum > 0:
            self.hsum_loss += _hsum_loss(w_transform)

        # l2norm of lowpass filter
        self.L2norm_loss = 0
        if self.lamL2norm > 0:
            self.L2norm_loss += _L2norm_loss(w_transform)

        # conjugate mirror filter condition
        self.CMF_loss = 0
        if self.lamCMF > 0:
            self.CMF_loss += _CMF_loss(w_transform)

        # convolution constraint
        self.conv_loss = 0
        if self.lamConv > 0:
            self.conv_loss += _conv_loss(w_transform)

        # L1 penalty on wavelet coeffs
        self.L1wave_loss = 0
        if self.lamL1wave > 0:
            self.L1wave_loss += _L1_wave_loss(data_t)

        # L1 penalty on attributions
        self.L1attr_loss = 0
        if self.lamL1attr > 0 and attributions is not None:
            self.L1attr_loss += _L1_attribution_loss(attributions)

        # Penalty on high frequency of h0  
        self.highfreq_loss = 0
        if self.lamHighfreq > 0:
            self.highfreq_loss += _penalty_high_freq(w_transform)

        # total loss
        loss = self.rec_loss + self.lamlSum * self.lsum_loss + self.lamhSum * self.hsum_loss + self.lamL2norm * self.L2norm_loss \
               + self.lamCMF * self.CMF_loss + self.lamConv * self.conv_loss + self.lamL1wave * self.L1wave_loss + self.lamL1attr * self.L1attr_loss \
               + self.lamHighfreq * self.highfreq_loss

        return loss


def _reconstruction_loss(data, recon_data):
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
        
    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size = recon_data.size(0)
    loss = F.mse_loss(recon_data, data, reduction="sum")
    loss = loss / batch_size

    return loss


def _lsum_loss(w_transform):
    """
    Calculate sum of lowpass filter
    """
    h0 = w_transform.h0
    loss = .5 * (h0.sum() - np.sqrt(2)) ** 2

    return loss


def _hsum_loss(w_transform):
    """
    Calculate sum of highpass filter
    """
    h0 = w_transform.h0
    h1 = low_to_high(h0)
    loss = .5 * h1.sum() ** 2

    return loss


def _L2norm_loss(w_transform):
    """
    Calculate L2 norm of lowpass filter
    """
    h0 = w_transform.h0
    loss = .5 * ((h0 ** 2).sum() - 1) ** 2

    return loss


def _CMF_loss(w_transform):
    """
    Calculate conjugate mirror filter condition
    """
    h0 = w_transform.h0
    n = h0.size(2)
    assert n % 2 == 0, "length of lowpass filter should be even"
    try:
        h_f = torch.fft.fft(torch.stack((h0, torch.zeros_like(h0)), dim=3), 1)
    except:
        h_f = torch.fft(torch.stack((h0, torch.zeros_like(h0)), dim=3), 1)
    mod = (h_f ** 2).sum(axis=3)
    cmf_identity = mod[0, 0, :n // 2] + mod[0, 0, n // 2:]
    loss = .5 * torch.sum((cmf_identity - 2) ** 2)

    return loss


def _conv_loss(w_transform):
    """
    Calculate convolution of lowpass filter
    """
    h0 = w_transform.h0
    n = h0.size(2)
    assert n % 2 == 0, "length of lowpass filter should be even"
    v = F.conv1d(h0, h0, stride=2, padding=n)
    e = torch.zeros_like(v)
    e[0, 0, n // 2] = 1
    loss = .5 * torch.sum((v - e) ** 2)

    return loss


def _L1_wave_loss(coeffs):
    """
    Calculate L1 norm of wavelet coefficients
    """
    batch_size = coeffs[0].size(0)
    loss = tuple_L1Loss(coeffs)
    loss = loss / batch_size

    return loss


def _L1_attribution_loss(attributions):
    """
    Calculate L1 norm of the attributions
    """
    batch_size = attributions[0].size(0)
    loss = tuple_L1Loss(attributions)
    loss = loss / batch_size

    return loss


def _penalty_high_freq(w_transform):
    # pen high frequency of h0
    n = w_transform.h0.size(2)
    h_f = torch.fft(torch.stack((w_transform.h0, torch.zeros_like(w_transform.h0)), dim=3), 1)
    mod = (h_f ** 2).sum(axis=3)
    left = int(np.floor(n / 4) + 1)
    right = int(np.ceil(3 * n / 4) - 1)
    h0_hf = mod[0, 0, left:right + 1]
    loss = 0.5 * torch.norm(h0_hf) ** 2

    return loss


def tuple_L1Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(abs(x[i]))
    return output / num


def tuple_L2Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(x[i] ** 2)
    return output / num
