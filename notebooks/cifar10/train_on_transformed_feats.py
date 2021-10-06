import numpy as np
import matplotlib.pyplot as plt
from awave.experimental.filters import gabor_filter, edge_filter, curve_filter
from awave.experimental.filters_agg import *
import awave.experimental.viz as viz
from tqdm import tqdm


import cifar10
from torch import nn
import torch
import torch.optim as optim
import util


class LinearClassifier(nn.Module):
    def __init__(self, input_size=10368, output_size=10):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        

    def forward(self, X):
        X = self.fc1(X)
        return X

if __name__ == '__main__':
    
    # specify the features
    W_conv2d0 = make_weights(7, [("color", i) for i in range(3)],
                             [("gabor", orientation,  offset)
                              for orientation in range(0, 180, 10)
                              for offset in [0, 7./8., 7./4, 7.*3/8.]])

    conv2d0 = nn.Conv2d(in_channels=3, out_channels=W_conv2d0.shape[-1], kernel_size=W_conv2d0.shape[0])
    conv2d0.weight.value = torch.Tensor(W_conv2d0.transpose())
    conv2d0.bias.value = 0
    pool2d0 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
    feat_extractor = nn.Sequential(conv2d0, pool2d0)

    # load data
    # this is too big for gpu
    print('loading data...')
    X, Y = cifar10.get_batch(batch_size=50000, train=True) # X is 1, 3, 32, 32
    X_test, Y_test = cifar10.get_batch(batch_size=10000, train=False) # X is 1, 3, 32, 32

    # extract feats
    print('extracting feats...')
    feats = feat_extractor(X)
    feats = feats.reshape(feats.shape[0], -1)
    feats_test = feat_extractor(X_test)
    feats_test = feats_test.reshape(feats_test.shape[0], -1)

    # set up dataloaders
    train_feats_loader  = cifar10.create_dataloader(feats, Y, batch_size=100)
    test_feats_loader = cifar10.create_dataloader(feats_test, Y_test, batch_size=100)


    # train
    print('training...')
    device = 'cuda'
    model = LinearClassifier(input_size=feats.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    util.train(model, device, train_feats_loader, test_feats_loader, optimizer, num_epochs, criterion)