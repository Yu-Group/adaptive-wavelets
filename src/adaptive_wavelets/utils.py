import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage.transform import rescale
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype) 


def compute_tuple_dim(x):
    tot_dim = 0
    for i in range(len(x)):
        shape = torch.tensor(x[i].shape)
        tot_dim += torch.prod(shape).item()
    return tot_dim


def add_noise(x, init_factor, noise_factor):
    '''add random noise to tensor
    Params
    ------
    x: torch.tensor
        input
    init_factor: float

    noise_factor: float
        amount of noise added to original filter
    '''    
    shape = x.shape
    x = init_factor*x + noise_factor*torch.randn(shape)
    return x


def pad_within(x, stride=2, start_row=0, start_col=0):
    w = x.new_zeros(stride, stride)
    if start_row == 0 and start_col == 0:
        w[0, 0] = 1
    elif start_row == 0 and start_col == 1:
        w[0, 1] = 1
    elif start_row == 1 and start_col == 0:
        w[1, 0] = 1
    else:
        w[1, 1] = 1
    if len(x.shape) == 2:
        x = x[None,None]
    return F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1)).squeeze()


def get_1dfilts(w_transform):
    '''Get 1d filters from one-dimensional wavelets
    Params
    ------
    w_transform: obj
        wavelet object
    '''    
    if w_transform.wt_type == 'DWT1d':
        h0 = F.pad(w_transform.h0.squeeze(), pad=(2,2), mode='constant', value=0)
        h1 = F.pad(w_transform.h1.squeeze(), pad=(2,2), mode='constant', value=0)      
        
        return (h0.detach().cpu(), h1.detach().cpu())
    
    else:
        raise ValueError('no such type of wavelet transform is supported')    


def get_2dfilts(w_transform):
    '''Get 2d filters from one-dimensional wavelets
    Params
    ------
    w_transform: obj
        wavelet object
    '''    
    if w_transform.wt_type == 'DTCWT2d':
        h0o = w_transform.xfm.h0o.data
        h1o = w_transform.xfm.h1o.data
        h0a = w_transform.xfm.h0a.data
        h1a = w_transform.xfm.h1a.data
        h0b = w_transform.xfm.h0b.data
        h1b = w_transform.xfm.h1b.data 
        
        # compute first level wavelet filters
        h0_r = F.pad(h0o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h0_i = F.pad(h0o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)
        h1_r = F.pad(h1o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h1_i = F.pad(h1o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)    

        lh_filt_r1 = h0_r.unsqueeze(0) * h1_r.unsqueeze(1)/np.sqrt(2)
        lh_filt_r2 = h0_i.unsqueeze(0) * h1_i.unsqueeze(1)/np.sqrt(2)
        lh_filt_i1 = h0_i.unsqueeze(0) * h1_r.unsqueeze(1)/np.sqrt(2)
        lh_filt_i2 = h0_r.unsqueeze(0) * h1_i.unsqueeze(1)/np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = h1_r.unsqueeze(0) * h1_r.unsqueeze(1)/np.sqrt(2)
        hh_filt_r2 = h1_i.unsqueeze(0) * h1_i.unsqueeze(1)/np.sqrt(2)
        hh_filt_i1 = h1_i.unsqueeze(0) * h1_r.unsqueeze(1)/np.sqrt(2)
        hh_filt_i2 = h1_r.unsqueeze(0) * h1_i.unsqueeze(1)/np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2 

        hl_filt_r1 = h1_r.unsqueeze(0) * h0_r.unsqueeze(1)/np.sqrt(2)
        hl_filt_r2 = h1_i.unsqueeze(0) * h0_i.unsqueeze(1)/np.sqrt(2)
        hl_filt_i1 = h1_i.unsqueeze(0) * h0_r.unsqueeze(1)/np.sqrt(2)
        hl_filt_i2 = h1_r.unsqueeze(0) * h0_i.unsqueeze(1)/np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2

        fl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        fl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]        
        
        # compute second level wavelet filters
        h0_a = h0a.squeeze().detach().cpu()
        h0_b = h0b.squeeze().detach().cpu()
        h1_a = h1a.squeeze().detach().cpu()
        h1_b = h1b.squeeze().detach().cpu()   
        
        lh_filt_r1 = pad_within(h0_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0)/np.sqrt(2)
        lh_filt_r2 = pad_within(h0_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1)/np.sqrt(2)
        lh_filt_i1 = pad_within(h0_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1)/np.sqrt(2)
        lh_filt_i2 = pad_within(h0_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0)/np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = pad_within(h1_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1)/np.sqrt(2)
        hh_filt_r2 = pad_within(h1_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0)/np.sqrt(2)
        hh_filt_i1 = pad_within(h1_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0)/np.sqrt(2)
        hh_filt_i2 = pad_within(h1_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1)/np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2 

        hl_filt_r1 = pad_within(h1_a.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=1)/np.sqrt(2)
        hl_filt_r2 = pad_within(h1_b.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=0)/np.sqrt(2)
        hl_filt_i1 = pad_within(h1_b.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=0)/np.sqrt(2)
        hl_filt_i2 = pad_within(h1_a.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=1)/np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2
        
        sl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        sl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]           
        
        return (fl_filt_reals, fl_filt_imags), (sl_filt_reals, sl_filt_imags)
    
    elif w_transform.wt_type == 'DWT2d':
        h0_col = F.pad(w_transform.h0.squeeze(), pad=(2,2), mode='constant', value=0)
        h1_col = F.pad(w_transform.h1.squeeze(), pad=(2,2), mode='constant', value=0)
        h0_row = F.pad(w_transform.h0.squeeze(), pad=(2,2), mode='constant', value=0)
        h1_row = F.pad(w_transform.h1.squeeze(), pad=(2,2), mode='constant', value=0)        
   
        filt_lh = h0_row.unsqueeze(0)*h1_col.unsqueeze(1)
        filt_hl = h1_row.unsqueeze(0)*h0_col.unsqueeze(1)
        filt_hh = h1_row.unsqueeze(0)*h1_col.unsqueeze(1)
        
        return (filt_lh.detach().cpu(), filt_hl.detach().cpu(), filt_hh.detach().cpu())
    
    else:
        raise ValueError('no such type of wavelet transform is supported')        
    

    
    
