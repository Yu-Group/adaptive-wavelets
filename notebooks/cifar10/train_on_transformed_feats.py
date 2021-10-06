import numpy as np
import matplotlib.pyplot as plt
from awave.experimental.filters import gabor_filter, edge_filter, curve_filter
from awave.experimental.filters_agg import *
import awave.experimental.viz as viz
from tqdm import tqdm
from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
import logging
import cifar10
from torch import nn
import torch
from torch.nn import functional as F
import torch.optim as optim
import util
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class LinearClassifier(LightningModule):
    def __init__(self, input_size=10368, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        

    def forward(self, X):
        X = self.fc1(X)
        return X
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss.item())
        preds = logits.softmax(dim=-1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log("val_acc", acc)
        return val_loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

if __name__ == '__main__':
    
    # specify the features
    W_conv2d0 = make_weights(7, [("color", i) for i in range(3)],
                             [("gabor", orientation,  offset)
                              for orientation in range(0, 180, 5)
                              for offset in [0, 7./8., 7./4, 7.*3/8.]])
    conv2d0 = nn.Conv2d(in_channels=3, out_channels=W_conv2d0.shape[-1], kernel_size=W_conv2d0.shape[0])
    conv2d0.weight.value = torch.Tensor(W_conv2d0.transpose())
    conv2d0.bias.value = 0
    pool2d0 = nn.MaxPool2d(kernel_size=5, stride=4, padding=0)
    feat_extractor = nn.Sequential(conv2d0, pool2d0)

    # load data
    # this is too big for gpu
    print('loading data...')
    X, Y = cifar10.get_batch(batch_size=50000, train=True) # X is 1, 3, 32, 32
    X_test, Y_test = cifar10.get_batch(batch_size=10000, train=False) # X is 1, 3, 32, 32

    # extract feats
    print('extracting feats...')
    with torch.no_grad():
        feats = feat_extractor(X).detach()
        feats = feats.reshape(feats.shape[0], -1)
        feats_test = feat_extractor(X_test).detach()
        feats_test = feats_test.reshape(feats_test.shape[0], -1)
        print('\tfeat shape', feats.shape)

    # set up dataloaders
    train_feats_loader  = cifar10.create_dataloader(feats, Y, batch_size=1000)
    test_feats_loader = cifar10.create_dataloader(feats_test, Y_test, batch_size=1000)

    # train
    print('training...')
    device = 'cuda'
    logger = CSVLogger("logs", name="my_exp_name")
    model = LinearClassifier(input_size=feats.shape[1]).to(device)
    trainer = Trainer(gpus=1, logger=logger, callbacks=[EarlyStopping(monitor="val_loss")])
    trainer.fit(model, train_feats_loader, test_feats_loader)