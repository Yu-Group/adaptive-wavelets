import numpy as np
import torch
import random
import os, sys
opj = os.path.join
import acd
from copy import deepcopy
sys.path.append('..')
sys.path.append('../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from numpy.fft import *
from torch import nn
import pickle as pkl
from transform_wrappers import *
from captum.attr import *
sys.path.append('../..')
from acd_wooseok.acd.scores import cd, score_funcs


def get_attributions(x_t: torch.Tensor, mt, class_num=0, device='cuda'):
    '''Returns all scores in a dict assuming mt works with both grads + CD

    Params
    ------
    mt: model
    class_num: target class
    '''
    x_t = x_t.to(device)
    mt.eval()
    mt = mt.to(device)

    results = {}
    attr_methods = ['IG', 'DeepLift', 'SHAP', 'CD', 'InputXGradient']
    for name, func in zip(attr_methods,
                          [IntegratedGradients, DeepLift, GradientShap, None, InputXGradient]):

        if name == 'CD':
            sweep_dim = 1
            tiles = acd.tiling_2d.gen_tiles(x_t[0,0,...,0], fill=0, method='cd', sweep_dim=sweep_dim)
            if x_t.shape[-1] == 2: # check for imaginary representations
                tiles = np.repeat(np.expand_dims(tiles, axis=-1), repeats=2, axis=3).squeeze()
            tiles = torch.Tensor(tiles).unsqueeze(1)
            attributions = score_funcs.get_scores_2d(mt, method='cd', ims=tiles, im_torch=x_t)[:, class_num].reshape(28,28)
        else:
            baseline = torch.zeros(x_t.shape).to(device)
            attributer = func(mt)
            if name in ['InputXGradient']:
                attributions = attributer.attribute(deepcopy(x_t), target=class_num)
            else:
                attributions = attributer.attribute(deepcopy(x_t), deepcopy(baseline), target=class_num)
            attributions = attributions.cpu().detach().numpy().squeeze()
            if x_t.shape[-1] == 2: # check for imaginary representations
                attributions = mag(attributions)
        results[name] = attributions
    return results


# test Net2c
def test_Net2c(test_loader, args, model, band_center=0.3, band_width=0.1):
    # eval mode
    model.eval()
    if args.cuda:
        model.cuda()

    # test
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data = bandpass_filter_augment(data, band_center, band_width)
        target = torch.zeros(2*args.test_batch_size, dtype=target.dtype)
        target[args.test_batch_size:] = 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= 2*len(test_loader.dataset)
    return test_loss, correct


def test_knockout_models(test_loader, args, true_band_centers, device='cuda'):
    true_band_num = len(true_band_centers)
    for i in range(true_band_num):
        # load model
        model = Net2c().to(device)
        model.load_state_dict(torch.load(opj('models/freq','net2c_' + str(i) + '.pth'), map_location=device))
        model = model.eval()

        # test accuracy of the model
        test_loss, correct = test_Net2c(test_loader, args, model, true_band_centers[i], band_width=0.1)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, 2*len(test_loader.dataset),
            100. * correct / (2*len(test_loader.dataset))))
