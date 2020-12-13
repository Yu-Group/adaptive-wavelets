import numpy as np
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from copy import deepcopy
from captum.attr import *
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class DTCWT_Transform(nn.Module):
    def __init__(self, biort='near_sym_b', qshift='qshift_b', J=5, device='cuda', requires_grad=True): 
        super(DTCWT_Transform, self).__init__()        
        self.xfm = DTCWTForward(biort=biort, qshift=qshift, J=J).to(device)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device)
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
        self.attributer = eval(attr_methods)(mt)     
        
    def forward(self, x: tuple, target=1):
        n = len(x)
        x_copy = []
        for i in range(n):
            x_copy.append(deepcopy(x[i].detach()))
        output_list = []
        for i in range(n):
            if self.attr_methods == 'InputXGradient':
                attributions = self.attributer.attribute(x[i], target=target, additional_forward_args=(x_copy, i))
            else:
                baseline = torch.zeros_like(x[i])
                attributions = self.attributer.attribute(x[i], baseline, target=target, additional_forward_args=(x_copy, i), n_steps=1)
            output_list.append(attributions)
        return tuple(output_list)            
            
            
class TrimModel(nn.Module):
    '''Prepends transformation onto network (with optional normalizaiton after the transform)
    Params
    '''
    def __init__(self, model, inv_transform, use_residuals=False):
        super(TrimModel, self).__init__()
        self.inv_transform = inv_transform
        self.model = model
        self.use_residuals = use_residuals

    def forward(self, s, t=None, indx=None, x_orig=None):
        '''
        Params
        ------
        s: torch.Tensor
            This should be the input in the transformed space which we want to interpret
            (batch_size, C, H, W) for images
            (batch_size, C, seq_length) for audio
        '''
        # untransform the input
        if t is None and indx is None:
            x = self.inv_transform(s)
        else:
            x = self.inv_transform(s, t, indx)
        
        # take residuals into account
        if self.use_residuals:
            assert x_orig is not None, "if using residuals, must also pass untransformed original image!"
            res = x_orig - x.detach()
            x = x + res

        # pass through the main model
        x = self.model.forward(x)
        return x            