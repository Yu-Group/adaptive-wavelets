import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../dsets/mnist')
import dset
from transforms_torch import bandpass_filter_augment
from tqdm import tqdm
from model import Net2c

def to_freq(x):
    x =  x.cpu().detach().numpy().squeeze()
    return np.fft.fftshift(mag(x))


def mag(x):
    '''Magnitude
    x[..., 0] is real part
    x[..., 1] is imag part
    '''
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]))
