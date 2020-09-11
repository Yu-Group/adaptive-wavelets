import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
from copy import deepcopy
from model_mnist import LeNet5
import dset_mnist as dset
sys.path.append('../trim')
from trim import *
from util import *
from attributions import *
from captum.attr import *

criterion_MSE = nn.MSELoss()
criterion_CE = nn.CrossEntropyLoss()
L1loss = nn.L1Loss()
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(data, model, trim_model, normalize=False):   
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    # recon loss
    s = model.transformer(inputs)
    recons = model.i_transformer(s)
    recon_loss = criterion_MSE(inputs, recons)
    # supervised loss
    outputs = trim_model.model(recons)
    supervised_loss = criterion_CE(outputs, labels)        
    # TRIM regularization
    attributer = InputXGradient(trim_model)
    s_ = deepcopy(s.detach())
    trim_loss = 0
    for label in range(10):            
        attributions = attributer.attribute(s_, target=label, additional_forward_args=deepcopy(inputs))
        if normalize is True:
            # standardization
            mean = attributions.mean(dim=1, keepdim=True)
            std = attributions.std(dim=1, keepdim=True)
            attributions = (attributions - mean) / std  
        trim_loss += L1loss(attributions, torch.zeros_like(attributions))     
        
    return(recon_loss, supervised_loss, trim_loss)
                           
                           
def train(epoch, train_loader, model, trim_model, lamb, normalize=False):
    model.train()
    train_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    ###################
    # train the model #
    ###################
    for i, data in enumerate(train_loader):
        # _ stands in for labels, here
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # calculate the loss
        recon_loss, supervised_loss, trim_loss = loss_function(data, model, trim_model, normalize=normalize)
        loss = supervised_loss + lamb * trim_loss           
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data[0].size(0)
        
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(data[0]), len(train_loader.dataset),
                   100. * i / len(train_loader), loss.data.item()), end='')        
    return(model, train_loss)                        


def test(test_loader, model, trim_model, lamb, normalize=False):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        # forward pass: compute predicted outputs by passing inputs to the model
        # calculate the loss
        recon_loss, supervised_loss, trim_loss = loss_function(data, model, trim_model, normalize=normalize)
        loss = supervised_loss + lamb * trim_loss           
        # update running training loss
        test_loss += loss.item() * data[0].size(0)    
    return(test_loss)                

