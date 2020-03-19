from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import fits
import torchvision.transforms.functional as F
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
classes = ['Mnu', 'Omegam', 'As', 'Sigma8', 'Unknown']

# Build PyTorch Dataset to read the fits files
class MassMapsDataset(Dataset):
    """Mass Maps Landmarks dataset."""

    def __init__(self, parameter_file, root_dir, transform=None, ncosmo=10):
        """
        Args:
            parameter_file (string): Path to the txt file with cosmological parameters.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ncosmo (int): Number of cosmologies to include in dataset
        """
        # m_nu, omega_m, A_s
        self.params = np.loadtxt(parameter_file)[:ncosmo]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.params)*10000 # We have 10,000 realisations for each cosmology

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,'model%03d/WLconv_z1.00_%04dr.fits'%(idx % len(self.params), idx // len(self.params)))
        image = fits.getdata(img_name)

        params = self.params[idx % len(self.params), 1:-1]
        
        sample = {'image': image, 'params': params}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, params = sample['image'], sample['params']
        return {'image': torch.from_numpy(image.reshape([1]+list(image.shape)).astype('float32') ),
                'params': torch.from_numpy(params.astype('float32'))}
    
    
# Build PyTorch Dataset to read the fits files
class MassMapsDatasetFilteredS8(Dataset):
    """Mass Maps Landmarks dataset. Only use sims with S8 within a certain range"""

    def __init__(self, parameter_file, root_dir, transform=None):
        """
        Args:
            parameter_file (string): Path to the txt file with cosmological parameters.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ncosmo (int): Number of cosmologies to include in dataset
        """
        self.params = np.loadtxt(parameter_file)[1:]
        om = self.params[:,2]
        sigma8 = self.params[:,4]
        S8 = sigma8*(om/0.3)**0.5
        mask = (S8 > 0.8 - 0.075) & (S8 < 0.8 + 0.075)
        self.index = np.where(mask)[0]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)*1000 # We have 1,000 realisations for each cosmology

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,'model%03d/WLconv_z1.00_%04dr.fits'%((self.index[idx % len(self.index)])+2, idx // len(self.index)))
        image = fits.getdata(img_name)

        params = self.params[self.index[idx % len(self.index)], [1, 2,-1]]
        
        sample = {'image': image, 'params': params}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class RandomToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, params = sample['image'], sample['params']
        if random.random() < 0.5:
            image = image[::-1]
        if random.random() < 0.5:
            image = image[:,::-1]
        if random.random() < 0.5:
            image = np.rot90(image)
        image = torch.from_numpy(image.reshape([1]+list(image.shape)).astype('float32'))
        params = torch.from_numpy(params.astype('float32'))
        image += torch.normal(image*0, image*0+0.001)
        return {'image': image,
                'params': params}