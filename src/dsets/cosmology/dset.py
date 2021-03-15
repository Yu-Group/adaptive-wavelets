from __future__ import print_function, division
import os, sys
opj = os.path.join
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import fits
import random
data_path = './data'

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
classes = ['Mnu', 'Omegam', 'As', 'Sigma8', 'Unknown']


def get_dataloader(root_dir, img_size=64, shuffle=True, split_train_test=True, pin_memory=True,
                   batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.   

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    dataset = MassMapsDatasetResized(root_dir, img_size)
    if split_train_test is True:
        train_loader = DataLoader(torch.utils.data.Subset(dataset, indices=range(20000)),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory,
                                  **kwargs)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, indices=range(20000, 23000)),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=pin_memory,
                                 **kwargs)
        return (train_loader, test_loader)
    else:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          pin_memory=pin_memory,
                          **kwargs)


# PyTorch 
class MassMapsDatasetResized(Dataset):
    """Mass Maps Landmarks dataset. Use sims with downsampled image"""

    def __init__(self, root_dir, img_size=64):
        """
        Args:
            root : string
                Root directory of dataset.
        """
        dataset_zip = np.load(opj(root_dir, 'cosmo_resize_{}.npz'.format(img_size)))
        self.imgs = dataset_zip['imgs']
        self.params = dataset_zip['params']

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self._ToTensor(self.imgs[idx])
        params = torch.from_numpy(self.params[idx].astype('float32'))
        return sample, params
        
        
    def _ToTensor(self, x):
        """Convert ndarrays to Tensors."""
        return torch.from_numpy(x.reshape([1]+list(x.shape)).astype('float32') )
