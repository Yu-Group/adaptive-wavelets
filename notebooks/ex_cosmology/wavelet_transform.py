import numpy as np
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from copy import deepcopy
from captum.attr import *
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse


class Wavelet_Transform(nn.Module):
    def __init__(self, wt_type='DTCWT', biort='near_sym_b', qshift='qshift_b', J=5, 
                 wave='db3', mode='zero', device='cuda', requires_grad=True): 
        super(Wavelet_Transform, self).__init__()     
        if wt_type == 'DTCWT':
            self.xfm = DTCWTForward(biort=biort, qshift=qshift, J=J).to(device)
            self.ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device)
        elif wt_type == 'DWT':
            self.xfm = DWTForward(wave=wave, J=J, mode=mode).to(device)
            self.ifm = DWTInverse(wave=wave, mode=mode).to(device)
        else: 
            raise ValueError('no such type of wavelet transform is supported')            
        self.J = J

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
    
    def tuple_sum(self, x):
        norm = 0
        num = len(x)
        for i in range(num):
            norm += torch.sum(x[i])
        return norm/num        
    

### TO DO!! ###
# incorporate mask for DWT
class DTCWT_Mask(nn.Module):
    def __init__(self, dtcwt, img_size=256, device='cuda'):
        '''
        dtcwt: DTCWT_Transform class
        '''
        super(DTCWT_Mask, self).__init__()
        self.img_size = img_size
        self.J = dtcwt.J

        # initialize masks 
        x = torch.randn(1,1,img_size,img_size).to(device)
        Y = dtcwt(x)   
        
        self.mask = nn.ParameterList()
        for i in range(self.J+1):
            self.mask.append(nn.Parameter(torch.ones_like(Y[i])))            
        
    def forward(self, x):
        output_list = []
        for i in range(self.J+1):
            output_list.append(torch.mul(x[i], self.mask[i]))
        return tuple(output_list)    
    
    def projection(self):
        for i in range(self.J+1):
            self.mask[i].data = torch.clamp(self.mask[i].data, 0, 1)    
            
            
class tuple_Attributer(nn.Module):
    def __init__(self, mt, attr_methods='InputXGradient', device='cuda'): 
        super(tuple_Attributer, self).__init__()
        self.mt = mt.to(device)
        self.attr_methods = attr_methods   
        self.device = device
        
    def forward(self, x: tuple, target=1):
        if self.attr_methods == 'InputXGradient':
            attributions = self.InputXGradient(x, target)
        elif self.attr_methods == 'IntegratedGradient':
            attributions = self.IntegratedGradient(x, target)
        else: 
            raise ValueError
        return attributions
        
    def InputXGradient(self, x: tuple, target=1):
        n = len(x)
        for i in range(n):
            x[i].retain_grad()
        output = self.mt(x)[0][target]
        output.backward(retain_graph=True)
        
        # input * gradient
        scores = []
        for i in range(n):
            scores.append(torch.mul(x[i], x[i].grad))
        return tuple(scores)  
    
    def IntegratedGradient(self, x: tuple, target=1, M=100):
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
        output = self.mt(input_vecs)[:,1].sum()
        output.backward(retain_graph=True)

        # ig
        scores = []
        for i in range(n):
            imps = input_vecs[i].grad.mean(0) * (x[i] - baselines[i]) # record all the grads
            scores.append(imps)   
        return tuple(scores)          
    
    
def create_images_high_attrs(attributions, im_t, i_transform, num_tot, num_seq=50):
    sp_levels = np.geomspace(1, num_tot, num_seq).astype(np.int)   
    device = 'cuda' if im_t[0].is_cuda else 'cpu'
    n = len(im_t)
    indx = 0
    results = []
    for i in range(num_seq):
        # sort attribution
        b = torch.tensor([])
        list_of_size = [0]

        for k in range(n):
            a = attributions[k].cpu().reshape(-1)
            b = torch.cat((b,a))
            list_of_size.append(list_of_size[-1] + a.shape[0])
        sort_order = torch.argsort(b, descending=True) 
        m = torch.zeros_like(b)
        m[sort_order[:sp_levels[indx]]] = 1

        list_of_masks = []
        for k in range(n):
            n0 = list_of_size[k]
            n1 = list_of_size[k+1]
            list_of_masks.append(m[n0:n1].reshape(im_t[k].shape))

        wm_list = []
        for k in range(n):
            wm_list.append(torch.mul(list_of_masks[k].to(device), im_t[k]))
        wm_list = tuple(wm_list)

        rec = i_transform(wm_list)       
        indx += 1
        results.append(rec.cpu())
    return results    
    
    
def compute_tuple_dim(x):
    tot_dim = 0
    for i in range(len(x)):
        shape = torch.tensor(x[i].shape)
        tot_dim += torch.prod(shape).item()
    return tot_dim

    
class Wavelet_Transform_from_Scratch(nn.Module):
    def __init__(self, init_wavelet='bior2.2', requires_grad=True, device='cuda'): 
        super(Wavelet_Transform_from_Scratch, self).__init__()        
        # initialize        
        w = pywt.Wavelet(init_wavelet)
        dec_hi = torch.tensor(w.dec_hi[::-1])
        dec_lo = torch.tensor(w.dec_lo[::-1])
        rec_hi = torch.tensor(w.rec_hi)
        rec_lo = torch.tensor(w.rec_lo)        
        self.dec_hi = nn.Parameter(dec_hi.to(device), requires_grad=requires_grad)
        self.dec_lo = nn.Parameter(dec_lo.to(device), requires_grad=requires_grad)
        self.rec_hi = nn.Parameter(rec_hi.to(device), requires_grad=requires_grad)
        self.rec_lo = nn.Parameter(rec_lo.to(device), requires_grad=requires_grad)
        
        self.filters = torch.stack([self.dec_lo.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                                    self.dec_lo.unsqueeze(0)*self.dec_hi.unsqueeze(1),
                                    self.dec_hi.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                                    self.dec_hi.unsqueeze(0)*self.dec_hi.unsqueeze(1)], dim=0)  
        
        self.inv_filters = torch.stack([self.rec_lo.unsqueeze(0)*self.rec_lo.unsqueeze(1),
                                        self.rec_lo.unsqueeze(0)*self.rec_hi.unsqueeze(1),
                                        self.rec_hi.unsqueeze(0)*self.rec_lo.unsqueeze(1),
                                        self.rec_hi.unsqueeze(0)*self.rec_hi.unsqueeze(1)], dim=0)        
    
    def xfm(self, x, levels=5):
        # wavelet transform
        h = x.size(2)
        w = x.size(3)
        x_t = F.conv2d(x, self.filters[:,None], stride=2, padding=2) # only works for filter of size 6
        res = x_t.clone()
        if levels > 1:
            res = self.xfm(res[:,:1], levels=levels-1)
            x_t[:,:1] = res
        x_t = x_t.reshape(-1,2,h//2,w//2).transpose(1,2).contiguous().reshape(-1,1,h,w)
        return x_t
    
    def ifm(self, x, levels=5):
        # inverse wavelet transform
        h = x.size(2)
        w = x.size(3)
        res = x.reshape(-1,h//2,2,w//2).transpose(1,2).contiguous().reshape(-1,4,h//2,w//2).clone()
        if levels > 1:
            res[:,:1] = self.ifm(res[:,:1], levels=levels-1)
        res = F.conv_transpose2d(res, self.inv_filters[:,None], stride=2)
        res = res[:,:,2:-2,2:-2]
        return res    