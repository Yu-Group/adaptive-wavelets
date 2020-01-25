import numpy as np
import torch
import acd
from copy import deepcopy
import sys
sys.path.append('../../dsets/mnist')
import dset
from util import *
from numpy.fft import *
from torch import nn
from style import *
from captum.attr import *
from transform_wrappers import *


def get_attributions(x_t: torch.Tensor, mt, class_num=1):
    '''Returns all scores in a dict assuming mt works with both grads + CD

    Params
    ------
    mt: model
    class_num: target class
    '''
    device_captum = 'cpu' # this only works with cpu
    x = x_t.unsqueeze(0).to(device_captum) # x is for the baseline
    x.requires_grad = True
    
    results = {}
    attr_methods = ['IG', 'DeepLift', 'SHAP', 'CD', 'InputXGradient']
    for name, func in zip(attr_methods,
                          [IntegratedGradients, DeepLift, GradientShap, None, Saliency, InputXGradient]):

        if name == 'CD':
            sweep_dim = 1
            tiles = acd.tiling_2d.gen_tiles(x_t.unsqueeze(0), fill=0, method='cd', sweep_dim=sweep_dim)
            if x_t.shape[-1] == 2: # check for imaginary representations
                tiles = np.repeat(np.expand_dims(tiles, axis=-1), repeats=2, axis=3).squeeze()
            attributions = acd.get_scores_2d(mt, method='cd', ims=tiles, im_torch=x_t)[:, class_num]
        else:
            baseline = torch.zeros(x.shape).to(device_captum)
            attributer = func(mt.to(device_captum))
            if name in ['InputXGradient']:
                attributions = attributer.attribute(deepcopy(x), target=class_num)
            else:
                attributions = attributer.attribute(deepcopy(x), deepcopy(baseline), target=class_num)
            attributions = attributions.cpu().detach().numpy().squeeze()
            if x_t.shape[-1] == 2: # check for imaginary representations
                attributions = mag(attributions)
        results[name] = attributions
    return results