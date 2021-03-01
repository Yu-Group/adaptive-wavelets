import torch.nn as nn
import pywt
import lowlevel
import torch
from utils import add_noise


def load_wavelet(wave: str, device=None):
    '''
    load 1-d wavelet from pywt allow both orthogonal and bi-orthogonal wavelets
    '''
    wave = pywt.Wavelet(wave)
    h0, h1 = wave.dec_lo, wave.dec_hi
    g0, g1 = wave.rec_lo, wave.rec_hi
    # Prepare the filters
    if h0[::-1] == g0 and h1[::-1] == g1:
        h0, h1 = lowlevel.prep_filt_afb1d(h0, h1, device)
        return (h0, h1)
    else:
        h0, h1 = lowlevel.prep_filt_afb1d(h0, h1, device)
        g0, g1 = lowlevel.prep_filt_sfb1d(g0, g1, device)
        return (h0, h1, g0, g1)

    
class DWT2d(nn.Module):
    '''Class of 2d wavelet transform 
    Params
    ------
    wt_type: str
        indicate either dual-tree complex wavelet transform (DTCWT) or discrete wavelet transform (DWT)
    biort: str
        one of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'. Specifies the first level biorthogonal wavelet filters. Can also
        give a two tuple for the low and highpass filters directly
    qshift: str
        one of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c', 'qshift_d'. Specifies the second level quarter shift filters. Can
            also give a 4-tuple for the low tree a, low tree b, high tree a and high tree b filters directly
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
    def __init__(self, wave='db3', mode='zero', J=5, init_factor=1, noise_factor=0):
        super().__init__() 
        wave = load_wavelet(wave)
        if len(wave) == 2:
            h0, h1 = wave[0], wave[1]
            g0, g1 = None, None
            self.is_bior = False
        elif len(wave) == 4:
            h0, h1, g0, g1 = wave[0], wave[1], wave[2], wave[3]   
            self.is_bior = True
        # prepare filts
        # initialize
        h0 = add_noise(h0, init_factor, noise_factor)
        h1 = add_noise(h1, init_factor, noise_factor)
        if self.is_bior:
            g0 = add_noise(g0, init_factor, noise_factor)
            g1 = add_noise(g1, init_factor, noise_factor)
        # parameterize
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.h1 = nn.Parameter(h1, requires_grad=True)
        if self.is_bior:
            self.g0 = nn.Parameter(g0, requires_grad=True)
            self.g1 = nn.Parameter(g1, requires_grad=True)
        
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
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)
        
        h0_col = self.h0.reshape((1, 1, -1, 1))
        h1_col = self.h1.reshape((1, 1, -1, 1))
        h0_row = self.h0.reshape((1, 1, 1, -1))
        h1_row = self.h1.reshape((1, 1, 1, -1))          

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.forward(
                ll, h0_col, h1_col, h0_row, h1_row, mode)
            yh.append(high)
            
        return tuple([ll]+[yh[i] for i in range(self.J)])
    
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
        
        if self.is_bior:
            g0_col = self.g0.reshape((1, 1, -1, 1))
            g1_col = self.g1.reshape((1, 1, -1, 1))
            g0_row = self.g0.reshape((1, 1, 1, -1))
            g1_row = self.g1.reshape((1, 1, 1, -1))              
        else:
            g0_col = self.h0.reshape((1, 1, -1, 1))
            g1_col = self.h1.reshape((1, 1, -1, 1))
            g0_row = self.h0.reshape((1, 1, 1, -1))
            g1_row = self.h1.reshape((1, 1, 1, -1))            

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = lowlevel.SFB2D.forward(
                ll, h, g0_col, g1_col, g0_row, g1_row, mode)
        return ll        
    