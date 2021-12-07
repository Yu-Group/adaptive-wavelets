import os

opj = os.path.join
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models

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


def get_validation(root_dir, img_size=64, pin_memory=True, batch_size=64, **kwargs):
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
    return DataLoader(torch.utils.data.Subset(dataset, indices=range(23000, 25000)),
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=pin_memory,
                      **kwargs)


def load_pretrained_model(model_name='resnet18', device='cuda', num_params=3, inplace=True,
                          data_path='/scratch/users/vision/data/cosmo'):
    '''Load a pretrained model and make shape alterations for cosmology
    '''

    # Modifying the model to predict the three cosmological parameters from single channel images
    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_params)

        if inplace == False:
            mods = list(model_ft.modules())
            for mod in mods:
                t = str(type(mod))
                if 'ReLU' in t:
                    mod.inplace = False

        model_ft = model_ft.to(device)
        if data_path is not None:
            model_ft.load_state_dict(torch.load(opj(data_path, 'resnet18_state_dict')))

    elif model_name == 'vgg16':
        model_ft = models.vgg16(pretrained=False)
        model_ft.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = 4096  # model_ft.fc.n_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load(opj(data_path, 'vgg16_adam_9_0.012')))

    model_ft.eval()
    # freeze layers
    for param in model_ft.parameters():
        param.requires_grad = False
    return model_ft


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
        return torch.from_numpy(x.reshape([1] + list(x.shape)).astype('float32'))
