import numpy as np
import matplotlib.pyplot as plt
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.ndimage import gaussian_filter
import os, sys
opj = os.path.join
from tqdm import tqdm
from functools import partial
import acd
from copy import deepcopy
sys.path.append('..')
sys.path.append('../..')
from transforms_torch import bandpass_filter, bandpass_filter_augment
# plt.style.use('dark_background')
sys.path.append('../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from numpy.fft import *
from torch import nn
import pickle as pkl

from acd_wooseok.acd.scores import cd

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

# set args
args = dset.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# load mnist data
train_loader, test_loader = dset.load_data(args.batch_size, args.test_batch_size, device)
test_num = len(test_loader.dataset.data)

# true band centers model is trained with
true_band_centers = np.linspace(0.15, 0.85, 20)
# number of band centers and test data
true_band_num = len(true_band_centers)

# test band centers
band_num = 120
band_centers = np.linspace(0.11, 0.89, band_num)

# record cd scores
scores_orig = torch.zeros(true_band_num, test_num, band_num) # cd score for class 0 (original img)
scores_filtered = torch.zeros(true_band_num, test_num, band_num) # cd score for class 1 (bandpass img)

for n_iter, true_band_center in enumerate(true_band_centers):
    if n_iter <= 17:
        pass
    else:
        # load model
        model = Net2c().to(device)
        model.load_state_dict(torch.load(opj('models/freq','net2c_' + str(n_iter) + '.pth'), map_location=device))
        model = model.eval()

        # test accuracy of the model
        # test_loss, correct = test_Net2c(test_loader, args, model, true_band_centers[band_center_idx], band_width=0.1)
        # test_losses.append(test_loss)
        # accuracies.append(correct.item())
        #
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, 2*len(test_loader.dataset),
        #     100. * correct / (2*len(test_loader.dataset))))
        print('\n number of iterations, true band center: %d, %d' %(n_iter, true_band_center))

        for batch_idx, (data, target) in enumerate(test_loader):
            batch_size = len(data)
            for data_idx in range(batch_size):
                im_torch = data[data_idx:data_idx+1]
                for band_idx, band_center in enumerate(band_centers):
                    # get cd score for each freqnecy band center
                    im_bandpass = im_torch.cpu()-bandpass_filter(im_torch, band_center)
                    score_orig = cd.cd(im_torch, model, mask=None, model_type='mnist', device='cuda',
                                               transform=partial(bandpass_filter, band_center=band_center))[0].flatten()
                    score_filtered = cd.cd(im_bandpass, model, mask=None, model_type='mnist', device='cuda',
                                               transform=partial(bandpass_filter, band_center=band_center))[0].flatten()
                    scores_orig[n_iter, data_idx+batch_idx*args.test_batch_size,band_idx] = score_orig[0].item()
                    scores_filtered[n_iter, data_idx+batch_idx*args.test_batch_size,band_idx] = score_filtered[1].item()
                    print('\r band index: {} [data {}th]'.format(band_idx, data_idx+batch_idx*args.test_batch_size), end='')

# save
pkl.dump([scores_orig, scores_filtered], open('./results/scores_cd_ver2-3', 'wb'))
