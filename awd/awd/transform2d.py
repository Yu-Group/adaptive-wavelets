import pywt
import torch
import torch.nn as nn
import torch.optim
import numpy as np

from awd.awd import lowlevel
from awd.awd.utils import init_filter, low_to_high
from awd.awd.losses import get_loss_f
from awd.awd.train import Trainer


class DWT2d(nn.Module):
    '''Class of 2d wavelet transform 
    Params
    ------
    J: int
        number of levels of decomposition
    wave: str
         which wavelet to use.
         can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
    mode: str
        'zero', 'symmetric', 'reflect' or 'periodization'. The padding scheme
    '''

    def __init__(self, wave='db3', mode='zero', J=5, init_factor=1, noise_factor=0, const_factor=0):
        super().__init__()
        h0, _ = lowlevel.load_wavelet(wave)
        
        # initialize
        h0 = init_filter(h0, init_factor, noise_factor, const_factor)
        
        # parameterize
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.J = J
        self.mode = mode
        self.wt_type = 'DWT2d'

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = ()
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        h1 = low_to_high(self.h0)
        h0_col = self.h0.reshape((1, 1, -1, 1))
        h1_col = h1.reshape((1, 1, -1, 1))
        h0_row = self.h0.reshape((1, 1, 1, -1))
        h1_row = h1.reshape((1, 1, 1, -1))

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.forward(
                ll, h0_col, h1_col, h0_row, h1_row, mode)
            yh += (high,)

        return (ll,) + yh

    def inverse(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        coeffs = list(coeffs)
        yl = coeffs.pop(0)
        yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        h1 = low_to_high(self.h0)
        g0_col = self.h0.reshape((1, 1, -1, 1))
        g1_col = h1.reshape((1, 1, -1, 1))
        g0_row = self.h0.reshape((1, 1, 1, -1))
        g1_row = h1.reshape((1, 1, 1, -1))

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB2D.forward(
                ll, h, g0_col, g1_col, g0_row, g1_row, mode)
        return ll
    
    def fit(self,
            X=None,
            train_loader=None,
            pretrained_model=None, 
            lr: float=0.001,
            num_epochs: int=20,
            seed: int=42,
            attr_methods = 'Saliency',
            target = 6,  
            lamlSum: float=1.,
            lamhSum: float=1.,
            lamL2norm: float=1.,
            lamCMF: float=1.,
            lamConv: float=1.,
            lamL1wave: float=1.,
            lamL1attr: float=1.,):
        """
        Params
        ------
        X: numpy array or torch.Tensor
        train_loader: data_loader
            each element should return tuple of (x, _)
        pretrained_model: nn.Module, optional
            pretrained model to distill
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
        torch.manual_seed(seed)
        if X is None and train_loader is None:
            raise ValueError('Either X or train_loader must be passed!')
        elif train_loader is None:
            if 'ndarray' in str(type(X)):
                device = 'cuda' if self.paramaters()[0].is_cuda else 'cpu'
                X = torch.Tensor(X).to(device)
            
            # convert to float
            X = X.float()
            X = X.unsqueeze(1)
            
            # need to pad as if it had y (to match default pytorch dataloaders)
            X = [(X[i], np.nan) for i in range(X.shape[0])]
            train_loader = torch.utils.data.DataLoader(X, 
                                                      shuffle=True,
                                                      batch_size=len(X))
#             print(iter(train_loader).next())
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_f = get_loss_f(lamlSum=lamlSum, lamhSum=lamhSum,
                            lamL2norm=lamL2norm, lamCMF=lamCMF, lamConv=lamConv,
                            lamL1wave=lamL1wave, lamL1attr=lamL1attr)
        trainer = Trainer(pretrained_model,
                          self,
                          optimizer,
                          loss_f,
                          use_residuals=True,
                          target=target,
                          attr_methods=attr_methods,
                          n_print=1)
        
        # actually train
        self.train()
        trainer(train_loader, epochs=num_epochs)
        self.train_losses = trainer.train_losses
        self.eval()


