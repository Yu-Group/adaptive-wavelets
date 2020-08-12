import numpy as np

import torch
from torch import nn

# DATASET and DATALOADER
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data


def get_dataloaders(n_samples_per_cluster,
                    latent_means,
                    latent_vars,     
                    extra_dim=8,
                    var=0.01,
                    batch_size=100, 
                    shuffle=True,
                    return_latents=False):
    """A generic data loader
    """
    latent_samples = generate_latent_samples(n_samples_per_cluster, latent_means, latent_vars)
    data = generate_full_samples(latent_samples, extra_dim, var) 
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(myDataset(data), batch_size=batch_size, shuffle=shuffle, **kwargs)  
    if return_latents:
        return data_loader, latent_samples
    else:
        return data_loader
    

def samples(mu, var, nb_samples=500):
    """
    Return a tensor of (nb_samples, features), sampled
    from the parameterized gaussian.
    :param mu: torch.Tensor of the means
    :param var: torch.Tensor of variances (NOTE: zero covars.)
    """
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt())
        ]
    return torch.stack(out, dim=0)


def generate_latent_samples(nb_samples, latent_means, latent_vars):
    latent_data = []
    n_clusters = len(latent_means)
    for i in range(n_clusters):
        cluster = samples(
            torch.Tensor(latent_means[i]),
            torch.Tensor(latent_vars[i]),
            nb_samples=nb_samples
        )
        latent_data.append(cluster)
        
    return torch.cat(latent_data, dim=0) 


def generate_full_samples(latent_samples, extra_dim, var):
    out = []
    nb_samples, dim = latent_samples.shape
    for i in range(nb_samples):
        zero = torch.zeros(extra_dim)
        mu = torch.cat([latent_samples[i], zero])
        v = var * torch.ones(dim + extra_dim)
        out += [
            torch.normal(mu, v.sqrt())
        ]
    return torch.stack(out, dim=0)    
    