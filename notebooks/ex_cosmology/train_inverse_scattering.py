import matplotlib.pyplot as plt
import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
from random import randint
from copy import deepcopy
import pickle as pkl
import argparse

from torch import nn
from models import AutoEncoder, AutoEncoderSimple, load_model
import torch.nn.functional as F

from models import load_model
sys.path.append('../../src')
sys.path.append('../../src/vae')
sys.path.append('../../src/vae/models')
sys.path.append('../../src/dsets/cosmology')
from dset import get_dataloader
from model import init_specific_model
from losses import get_loss_f, _reconstruction_loss
from training import Trainer
from viz import viz_im_r, cshow, viz_filters
from sim_cosmology import p

sys.path.append('../../lib/trim')
# trim modules
from trim import DecoderEncoder, TrimModel
from captum.attr import *

# wavelet
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from kymatio.torch import Scattering2D
from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2
from colorsys import hls_to_rgb


class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2

        self.main = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_input_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),            

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)
    
    
# create dir
out_dir = opj(p.out_dir, p.dirname)
os.makedirs(out_dir, exist_ok=True)  

# seed
random.seed(p.seed)
np.random.seed(p.seed)
torch.manual_seed(p.seed)

# get dataloaders
train_loader = get_dataloader(p.data_path, 
                              img_size=256,
                              batch_size=p.train_batch_size)
im = iter(train_loader).next()[0][0:1].to(device)

# load model
model = load_model(model_name='resnet18', device=device, data_path=p.data_path)
model = model.eval()
# freeze layers
for param in model.parameters():
    param.requires_grad = False
    
    
# scatter transform
M = 256
J = 3
L = 8
scattering = Scattering2D(J, shape=(M, M), L=L, max_order=2).to(device)
shape = scattering(im).shape[2:]

num_input_channels = shape[0]
num_hidden_channels = num_input_channels


generator = Generator(num_input_channels, num_hidden_channels).to(device)
generator.train()

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)

# Training Loop
# Lists to keep track of progress
losses = []
num_epochs = 100
            
for epoch in range(num_epochs):
    print('Training epoch {}\n'.format(epoch))
    epoch_loss = 0
    for batch_idx, (current_batch, _) in enumerate(train_loader):
        generator.zero_grad()
        batch_images = current_batch.to(device)
        batch_scattering = scattering(batch_images).squeeze(1)
        batch_inverse_scattering = generator(batch_scattering)
        loss = criterion(batch_inverse_scattering, batch_images)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.data.item()
        
        # Output training stats
        if batch_idx % 50 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(current_batch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()), end='') 

            # SAVE MODEL
            torch.save(generator.state_dict(), 'results/inverse_scatter_transform_max_order=1.pth')          

    # Save Losses for plotting later
    losses.append(epoch_loss/(batch_idx + 1))
    
pkl.dump(losses, open('results/inverse_scatter_transform_max_order=1.pkl', 'wb'))
    
      