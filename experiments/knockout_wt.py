import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os, sys
opj = os.path.join
from tqdm import tqdm
from functools import partial
import acd
from copy import deepcopy
sys.path.append('..')
from transforms_torch import perturb_wt, perturb_wt_augment
import transform_wrappers
sys.path.append('../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from torch import nn
from style import *
from pytorch_wavelets import DWTForward, DWTInverse


def train_Net2c(train_loader, args, t, transform_i, save_path='models/cnn_vanilla.pth'):
    # seed
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    if device == 'cuda':
        torch.cuda.manual_seed(13)

    # model
    model = Net2c().to(device)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # train
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            B = len(target)
            data = perturb_wt_augment(data, t, transform_i).to(device)
            target = torch.ones(2*B, dtype=target.dtype).to(device)
            target[B:] = 0
            # zero grad
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * 2*B, 2*len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()), end='')

    torch.save(model.state_dict(), save_path)
    
    
# test Net2c
def test_Net2c(test_loader, model, t, transform_i):
    # eval mode
    model = model.to(device)
    model.eval()

    # test
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        B = len(target)
        data = perturb_wt_augment(data, t, transform_i).to(device)
        target = torch.ones(2*B, dtype=target.dtype).to(device)
        target[B:] = 0
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= 2*len(test_loader.dataset)
    return test_loss, correct    

if __name__ == '__main__':
    # set args
    args = dset.get_args()

    # load mnist data
    train_loader, test_loader = dset.load_data(args.batch_size, args.test_batch_size, device)
    
    # wavelet transform
    xfm = DWTForward(J=3, mode='symmetric', wave='db4')
    ifm = DWTInverse(mode='symmetric', wave='db4')
    t = lambda x: xfm(x)
    transform_i = lambda x: ifm(x)    

    # train model
    train_Net2c(train_loader, args, t, transform_i, save_path=opj('mnist/models/wt','net2c_' + str(0) + '.pth'))
    model = Net2c().to(device)
    model.load_state_dict(torch.load(opj('mnist/models/wt','net2c_' + str(0) + '.pth'), map_location=device))
    
    # test model
    test_loss, correct = test_Net2c(test_loader, model, t, transform_i)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, 2*len(test_loader.dataset),
        100. * correct / (2*len(test_loader.dataset))))    
