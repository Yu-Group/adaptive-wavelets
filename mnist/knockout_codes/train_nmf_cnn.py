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
sys.path.append('../..')
sys.path.append('../../..')
from transforms_torch import bandpass_filter
# plt.style.use('dark_background')
sys.path.append('../../../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from numpy.fft import *
from torch import nn
from style import *
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
import pickle as pkl
from torchvision import datasets, transforms
from sklearn.decomposition import NMF
import transform_wrappers
import visualize as viz
from model import Net, Net2c
torch.manual_seed(42)
np.random.seed(42)

from acd_wooseok.acd.scores import cd
from acd_wooseok.acd.util import tiling_2d
from acd_wooseok.acd.scores import score_funcs
from torchvision import datasets, transforms
# import modules
from funcs import *
from matfac import *

# load args
args = dset.get_args()
args.batch_size = int(args.batch_size/2) # half the batchsize
args.epochs = 50
args.cuda = not args.no_cuda and torch.cuda.is_available()

# load mnist dataloader
train_loader, test_loader = dset.load_data_with_indices(args.batch_size, args.test_batch_size, device)

# dataset
X = train_loader.dataset.data.numpy().astype(np.float32)
X = X.reshape(X.shape[0], -1)
X /= 255
Y = train_loader.dataset.targets.numpy()

X_test = test_loader.dataset.data.numpy().astype(np.float32)
X_test = X_test.reshape(X_test.shape[0], -1)
X_test /= 255
Y_test = test_loader.dataset.targets.numpy()

# load NMF object
# run NMF
# nmf = NMF(n_components=30, max_iter=1000)
# nmf.fit(X)
# pkl.dump(nmf, open('./results/nmf_30.pkl', 'wb'))
nmf = pkl.load(open('../results/nmf_30.pkl', 'rb'))
D = nmf.components_
# nmf transform
W = nmf.transform(X)
W_test = nmf.transform(X_test)


def nmf_transform(W: np.array, data_indx, list_dict_indx=[0]):
    im_parts = W[data_indx][:,list_dict_indx] @ D[list_dict_indx] / 0.3081
    im_parts = torch.Tensor(im_parts).reshape(batch_size, 1, 28, 28)
    return im_parts

def nmf_knockout_augment(im: torch.Tensor, W: np.array, data_indx, list_dict_indx=[0]):
    batch_size = im.size()[0]
    im_copy = deepcopy(im)
    im_parts = nmf_transform(W, data_indx, list_dict_indx)
    im_copy = torch.cat((im_copy,im-im_parts), dim=0)
    return im_copy


for dict_indx in range(nmf.n_components_):
    # knockout first dictionary and redefine train and test dataset
    indx = np.argwhere(W[:,dict_indx] > 0).flatten()
    indx_t = np.argwhere(W_test[:,dict_indx] > 0).flatten()

    # subset dataloader
    train_loader, test_loader = dset.load_data_with_indices(args.batch_size,
                                                            args.test_batch_size,
                                                            device,
                                                            subset_index=[indx, indx_t])

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
        for batch_indx, (data, target, data_indx) in enumerate(train_loader):
            batch_size = len(data)
            data = nmf_knockout_augment(data, W, data_indx, list_dict_indx=[dict_indx])
            target = torch.zeros(2*batch_size, dtype=target.dtype)
            target[batch_size:] = 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_indx % args.log_interval == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_indx * len(data), 2*len(train_loader.dataset),
                           100. * batch_indx / len(train_loader), loss.data.item()), end='')

    torch.save(model.state_dict(), '../models/nmf/net2c_{}.pth'.format(dict_indx))

    # eval mode
    model.eval()
    if args.cuda:
        model.cuda()

    # test
    test_loss = 0
    correct = 0
    for batch_indx, (data, target, data_indx) in tqdm(enumerate(test_loader)):
        batch_size = len(data)
        data = nmf_knockout_augment(data, W_test, data_indx, list_dict_indx=[dict_indx])
        target = torch.zeros(2*batch_size, dtype=target.dtype)
        target[batch_size:] = 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= 2*len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 2*len(test_loader.dataset),
        100. * correct / (2*len(test_loader.dataset))))
