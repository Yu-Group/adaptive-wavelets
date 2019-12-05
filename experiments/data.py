from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import fits

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Build PyTorch Dataset to read the fits files
class MassMapsDataset(Dataset):
    """Mass Maps Landmarks dataset."""

    def __init__(self, parameter_file, root_dir, transform=None, ncosmo=3):
        """
        Args:
            parameter_file (string): Path to the txt file with cosmological parameters.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ncosmo (int): Number of cosmologies to include in dataset
        """
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