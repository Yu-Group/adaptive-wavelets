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

def visualize(im_orig, transform):
    plt.figure(dpi=100, figsize=(9, 6))
    R, C = 2, 3
    tits = ['orig', 'transformed', 'diff']
    ims = [im_orig, transform(im_orig), im_orig - transform(im_orig)]
    for i in range(3):    
        plt.subplot(R, C, i + 1)
        plt.title(tits[i])
        plt.imshow(ims[i])
        plt.axis('off')
    
    for i in range(3):
        plt.subplot(R, C, 4 + i)
        plt.imshow(np.fft.fftshift(np.abs(np.fft.fft2(ims[i]))))
        plt.xticks([0, im_orig.shape[0] / 2, im_orig.shape[0] - 1], labels=[-1, 0, 1])
        plt.yticks([0, im_orig.shape[1] / 2, im_orig.shape[1] - 1], labels=[-1, 0, 1])
        plt.xlabel('frequency x')
        plt.ylabel('frequency y')

    plt.tight_layout()
    
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
    
