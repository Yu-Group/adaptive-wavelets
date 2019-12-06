import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import fits
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.ndimage import gaussian_filter
from os.path import join as oj
import sys
from tqdm import tqdm
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from data import *


def train_model(model, criterion, optimizer, scheduler, num_epochs=35):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            

            # Iterate over data.
            i = 0
            for data in dataloader:
                inputs, params = data['image'], data['params']
                inputs = inputs.to(device)
                params = params.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, params)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                i += 1
                if i > 100:
                    break
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (len(mnu_dataset) / 100 * 16)
            
            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))
            
            
            # Saving fairly well trained model
            torch.save(model_ft.state_dict(), oj(data_path, f'vgg16_adam_{epoch}_{epoch_loss:0.3f}'))

            # # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    data_path = '/scratch/users/vision/data/cosmo'
    mnu_dataset = MassMapsDataset(oj(data_path, 'cosmological_parameters.txt'),  oj(data_path, 'z1_256'))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_ft = models.resnet18(pretrained=False)
    model_ft = models.vgg16(pretrained=False)

    # Modifying the model to predict the three cosmological parameters from single channel images
    # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 3)
    # model_ft = model_ft.to(device)

    # Modifying the model to predict the three cosmological parameters from single channel images
    model_ft.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    num_ftrs = 4096 # model_ft.fc.n_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)


    data_transform = transforms.Compose([
            ToTensor()
        ])
    mnu_dataset = MassMapsDataset(oj(data_path, 'cosmological_parameters.txt'),  
                                  oj(data_path, 'z1_256'),
                                  transform=data_transform)
    dataloader = torch.utils.data.DataLoader(mnu_dataset, batch_size=16, 
                                             shuffle=True, num_workers=1)


    criterion = torch.nn.L1Loss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)

    # model_ft = model_ft.load_state_dict()

