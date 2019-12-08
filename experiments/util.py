import numpy as np
import matplotlib.pyplot as plt
import foolbox
import torch
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.ndimage import gaussian_filter

import sys
sys.path.append('../visualization')
import bandpass_filter


    
class Transforms:
    def __init__(self):
        self.sigma = 1
    def gaussian_filter(self, x): 
        '''Filter with given sigma
        '''
        if 'Tensor' in str(type(x)):
            x = x.cpu().numpy().squeeze()
            return torch.Tensor(gaussian_filter(x, sigma=self.sigma)).reshape(1, 1, 28 , 28)
        return gaussian_filter(x, sigma=self.sigma)
    
    def bandpass_filter(self, x, delta=0.05):
        '''Filter with freq band between low and high
        Low and high should be between 0.1 and 0.9 (not inclusive)
        '''
        low = self.sigma - delta
        high = self.sigma + delta
        if 'Tensor' in str(type(x)):
            x = x.cpu().numpy().squeeze()
            out = bandpass_filter.bandpass_filter(x, cutoff_low=low, cutoff_high=high)
            return torch.Tensor(out) #.reshape(1, 1, 28, 28)
        return bandpass_filter.bandpass_filter(x, cutoff_low=low, cutoff_high=high)
    
    def bandpass_filter_approx(self, x, delta=0.25): 
        '''Filter with within freq band near self.sigma of plus or minus 0.1
        N.B. this bandpass filter is kind of hacky and could be better
        '''
        if 'Tensor' in str(type(x)):
            x = x.cpu().numpy().squeeze()
            lower_freqs = gaussian_filter(x, sigma=self.sigma + delta)
            higher_freqs = gaussian_filter(x, sigma=self.sigma - delta)
            out = higher_freqs - lower_freqs
            return torch.Tensor(out).reshape(1, 1, 28, 28)
        else:
            lower_freqs = gaussian_filter(x, sigma=self.sigma + delta)
            higher_freqs = gaussian_filter(x, sigma=self.sigma - delta)
            out = higher_freqs - lower_freqs            
            return out  
    
