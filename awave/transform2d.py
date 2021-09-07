import torch
import torch.nn as nn
import torch.optim

from awave.utils import lowlevel
from awave.utils.misc import init_filter, low_to_high
from awave.transform import AbstractWT

class DWT2d(AbstractWT):
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


