import numpy as np
import matplotlib.pyplot as plt
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.ndimage import gaussian_filter
import sys
from tqdm import tqdm
from functools import partial
import acd
from copy import deepcopy
sys.path.append('..')
from transforms_torch import bandpass_filter, bandpass_filter_augment
plt.style.use('dark_background')
sys.path.append('../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from numpy.fft import *
from torch import nn
import os
opj = os.path.join

# train Net2c for classifying original images vs filtered images
def train_Net2c(train_loader, args, band_center=0.3, band_width=0.1, save_path='models/cnn_vanilla.pth'):
    # set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # create model
    model = Net2c()
    if args.cuda:
        model.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # train
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = bandpass_filter_augment(data, band_center, band_width)
            target = torch.zeros(2*args.batch_size, dtype=target.dtype)
            target[args.batch_size:] = 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 2*len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()), end='')

    torch.save(model.state_dict(), save_path)

# set args
args = dset.get_args()
args.batch_size = int(args.batch_size/2) # half the batchsize
args.cuda = not args.no_cuda and torch.cuda.is_available()

# load mnist data
train_loader, _ = dset.load_data(args.batch_size, args.test_batch_size, args.cuda)

# train model
band_centers = np.linspace(0.15, 0.85, 20)
for i, band_center in tqdm(enumerate(band_centers)):
    train_Net2c(train_loader, args, band_center=band_center, band_width=0.1, save_path=opj('models','net2c_' + str(i) + '.pth'))
