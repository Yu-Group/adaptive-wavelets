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
from transforms_torch import bandpass_filter, bandpass_filter_augment, batch_fftshift2d, batch_ifftshift2d
import transform_wrappers
sys.path.append('../dsets/mnist')
import dset
from model import Net, Net2c
from util import *
from torch import nn
from style import *


def train_Net2c(train_loader, args, band_center=0.3, band_width=0.1, save_path='models/cnn_vanilla.pth'):
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
            data = bandpass_filter_augment(data, band_center, band_width).to(device)
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
def test_Net2c(test_loader, model, band_center=0.3, band_width=0.1):
    # eval mode
    model = model.to(device)
    model.eval()

    # test
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        B = len(target)
        data = bandpass_filter_augment(data, band_center, band_width).to(device)
        target = torch.ones(2*B, dtype=target.dtype).to(device)
        target[B:] = 0
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


if __name__ == '__main__':
    # set args
    args = dset.get_args()

    # load mnist data
    train_loader, test_loader = dset.load_data(args.batch_size, args.test_batch_size, device)
    
    # freq band
    band_center = 0.3
    band_width = 0.05

    # train model
    train_Net2c(train_loader, args, band_center, band_width, save_path=opj('mnist/models/freq','net2c_' + str(0) + '.pth'))
    model = Net2c().to(device)
    model.load_state_dict(torch.load(opj('mnist/models/freq','net2c_' + str(0) + '.pth'), map_location=device))
    
    # test model
    test_loss, correct = test_Net2c(test_loader, model, band_center, band_width)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, 2*len(test_loader.dataset),
        100. * correct / (2*len(test_loader.dataset))))  