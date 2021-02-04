import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from copy import deepcopy
from captum.attr import *
from utils import pad_within
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse
import pywt


class Wavelet_Transform(nn.Module):
    '''Class of wavelet transform 
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
    device: str
        use GPU or CPU
    '''
    def __init__(self, wt_type='DTCWT', biort='near_sym_b', qshift='qshift_b', J=5, 
                 wave='db3', mode='zero', device='cuda', requires_grad=True): 
        super().__init__()     
        if wt_type == 'DTCWT':
            self.xfm = DTCWTForward(biort=biort, qshift=qshift, J=J).to(device)
            self.ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device)
        elif wt_type == 'DWT':
            self.xfm = DWTForward(wave=wave, J=J, mode=mode).to(device)
            self.ifm = DWTInverse(wave=wave, mode=mode).to(device)
        else: 
            raise ValueError('no such type of wavelet transform is supported')            
        self.J = J
        self.wt_type = wt_type

    def forward(self, x):
        # forward wavelet transform
        Yl, Yh = self.xfm(x)
        output_list = [Yl]
        for i in range(self.J):
            output_list.append(Yh[i])        
        return tuple(output_list)
    
    def inverse(self, x):
        # inverse wavelet transform
        x = list(x)
        Yl = x.pop(0)
        Yh = x
        return self.ifm((Yl, Yh))      
    
    
class Attributer(nn.Module):
    '''Get attribution scores for wavelet coefficients
    Params
    ------
    mt: nn.Module
        model after all the transformations
    attr_methods: str
        currently support InputXGradient only
    device: str
        use GPU or CPU
    '''    
    def __init__(self, mt, attr_methods='InputXGradient', device='cuda'): 
        super().__init__()
        self.mt = mt.to(device)
        self.attr_methods = attr_methods   
        self.device = device
        
    def forward(self, x: tuple, target=1, additional_forward_args=None):
        if self.attr_methods == 'InputXGradient':
            attributions = self.InputXGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'IntegratedGradient':
            attributions = self.IntegratedGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'Saliency':
            attributions = self.Saliency(x, target, additional_forward_args)
        else: 
            raise ValueError
        return attributions
        
    def InputXGradient(self, x: tuple, target=1, additional_forward_args=None):
        outputs = self.mt(x, additional_forward_args)[:,target]
        grads = torch.autograd.grad(torch.unbind(outputs), x)        
        # input * gradient
        attributions = tuple(xi * gi for xi, gi in zip(x, grads))
        return attributions    
    
    def Saliency(self, x: tuple, target=1, additional_forward_args=None):
        outputs = self.mt(x, additional_forward_args)[:,target]
        grads = torch.autograd.grad(torch.unbind(outputs), x)        
        return grads
    
    ### TO DO!! ###
    # implement batch version of IG
    def IntegratedGradient(self, x: tuple, target=1, additional_forward_args=None, M=100):
        n = len(x)
        mult_grid = np.array(range(M))/(M-1) # fractions to multiply by

        # compute all the input vecs
        input_vecs = []
        baselines = []
        for i in range(n):
            baselines.append(torch.zeros_like(x[i])) # baseline of zeros
            shape = list(x[i].shape[1:])
            shape.insert(0, M)
            inp = torch.empty(shape, dtype=torch.float32, requires_grad=True).to(self.device)    
            for j, prop in enumerate(mult_grid):
                inp[j] = baselines[i] + prop * (x[i] - baselines[i])
            inp.retain_grad()
            input_vecs.append(inp)

        # run forward pass
        output = self.mt(input_vecs, additional_forward_args)[:,1].sum()
        output.backward(retain_graph=True)

        # ig
        scores = []
        for i in range(n):
            imps = input_vecs[i].grad.mean(0) * (x[i] - baselines[i]) # record all the grads
            scores.append(imps)   
        return tuple(scores)    
    
    
def initialize_filters(w_transform, init_level=1, noise_level=0.1, device=device):
    '''Initialize wavelet filters by adding random noise
    Params
    ------
    w_transform: obj
        wavelet object
    noise_level: float
        amount of noise added to original filter
    '''        
    w_t = deepcopy(w_transform)
    if w_transform.wt_type == 'DTCWT':
        w_t.xfm.h0o.data = init_level*w_transform.xfm.h0o.data + noise_level*torch.randn(w_transform.xfm.h0o.data.shape).to(device)
        w_t.xfm.h1o.data = init_level*w_transform.xfm.h1o.data + noise_level*torch.randn(w_transform.xfm.h1o.data.shape).to(device)
        w_t.xfm.h0a.data = init_level*w_transform.xfm.h0a.data + noise_level*torch.randn(w_transform.xfm.h0a.data.shape).to(device)
        w_t.xfm.h1a.data = init_level*w_transform.xfm.h1a.data + noise_level*torch.randn(w_transform.xfm.h1a.data.shape).to(device)
        w_t.xfm.h0b.data = init_level*w_transform.xfm.h0b.data + noise_level*torch.randn(w_transform.xfm.h0b.data.shape).to(device)
        w_t.xfm.h1b.data = init_level*w_transform.xfm.h1b.data + noise_level*torch.randn(w_transform.xfm.h1b.data.shape).to(device)
        w_t.ifm.g0o.data = init_level*w_transform.ifm.g0o.data + noise_level*torch.randn(w_transform.ifm.g0o.data.shape).to(device)
        w_t.ifm.g1o.data = init_level*w_transform.ifm.g1o.data + noise_level*torch.randn(w_transform.ifm.g1o.data.shape).to(device)
        w_t.ifm.g0a.data = init_level*w_transform.ifm.g0a.data + noise_level*torch.randn(w_transform.ifm.g0a.data.shape).to(device)
        w_t.ifm.g1a.data = init_level*w_transform.ifm.g1a.data + noise_level*torch.randn(w_transform.ifm.g1a.data.shape).to(device)
        w_t.ifm.g0b.data = init_level*w_transform.ifm.g0b.data + noise_level*torch.randn(w_transform.ifm.g0b.data.shape).to(device)
        w_t.ifm.g1b.data = init_level*w_transform.ifm.g1b.data + noise_level*torch.randn(w_transform.ifm.g1b.data.shape).to(device)
    elif w_transform.wt_type == 'DWT':
        w_t.xfm.h0_row.data = init_level*w_transform.xfm.h0_row.data + noise_level*torch.randn(w_transform.xfm.h0_row.data.shape).to(device)
        w_t.xfm.h1_row.data = init_level*w_transform.xfm.h1_row.data + noise_level*torch.randn(w_transform.xfm.h1_row.data.shape).to(device)
        w_t.xfm.h0_col.data = init_level*w_transform.xfm.h0_col.data + noise_level*torch.randn(w_transform.xfm.h0_col.data.shape).to(device)
        w_t.xfm.h1_col.data = init_level*w_transform.xfm.h1_col.data + noise_level*torch.randn(w_transform.xfm.h1_col.data.shape).to(device)
        w_t.ifm.g0_row.data = init_level*w_transform.ifm.g0_row.data + noise_level*torch.randn(w_transform.ifm.g0_row.data.shape).to(device)
        w_t.ifm.g1_row.data = init_level*w_transform.ifm.g1_row.data + noise_level*torch.randn(w_transform.ifm.g1_row.data.shape).to(device)
        w_t.ifm.g0_col.data = init_level*w_transform.ifm.g0_col.data + noise_level*torch.randn(w_transform.ifm.g0_col.data.shape).to(device)
        w_t.ifm.g1_col.data = init_level*w_transform.ifm.g1_col.data + noise_level*torch.randn(w_transform.ifm.g1_col.data.shape).to(device)        
    else: 
        raise ValueError('no such type of wavelet transform is supported')            
    return w_t    
    
    
def get_2dfilts(w_transform):
    ### TO DO: implement 2d filters for inverse transform ###
    '''Get 2d filters from one-dimensional wavelets
    Params
    ------
    w_transform: obj
        wavelet object
    wt_type: str
        indicate either dual-tree complex wavelet transform (DTCWT) or discrete wavelet transform (DWT)
    '''    
    if w_transform.wt_type == 'DTCWT':
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
    
    elif w_transform.wt_type == 'DWT':
        h0_row = F.pad(w_transform.xfm.h0_row.squeeze(), pad=(2,2), mode='constant', value=0)
        h1_row = F.pad(w_transform.xfm.h1_row.squeeze(), pad=(2,2), mode='constant', value=0)        
        h0_col = F.pad(w_transform.xfm.h0_col.squeeze(), pad=(2,2), mode='constant', value=0)
        h1_col = F.pad(w_transform.xfm.h1_col.squeeze(), pad=(2,2), mode='constant', value=0)         
        
        filt_lh = h0_row.unsqueeze(0)*h1_col.unsqueeze(1)
        filt_hl = h1_row.unsqueeze(0)*h0_col.unsqueeze(1)
        filt_hh = h1_row.unsqueeze(0)*h1_col.unsqueeze(1)
        
        return (filt_lh.detach().cpu(), filt_hl.detach().cpu(), filt_hh.detach().cpu())
    
    else:
        raise ValueError('no such type of wavelet transform is supported')        
    


