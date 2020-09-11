import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from scipy import stats
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_2d_samples(sample, color='C0'):
    """Plot 2d sample 
    
        Arugments
        ---------
        sample : 2D ndarray or tensor
            matrix of spatial coordinates for each sample       
    """
    if "torch" in str(type(sample)):
        sample_np = sample.detach().cpu().numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y, color=color)
    plt.gca().set_aspect('equal', adjustable='box')
    
    
def plot_2d_latent_samples(latent_sample, color='C0'):
    """Plot latent samples select two most highly variable coordinates
    
        Arugments
        ---------
        latent_sample : tensor
            matrix of spatial coordinates for each latent sample       
    """    
    latent_dim = latent_sample.size()[1]
    stds = []
    for i in range(latent_dim):
        stds.append(torch.std(latent_sample[:,i]).item())
    stds = np.array(stds)
    ind = np.argsort(stds)[::-1][:2]
    plot_2d_samples(latent_sample[:,list(ind)])    
    
    
def traverse_line(idx, model, n_samples=100, n_latents=2, data=None, max_traversal=10):
    """Return a (size, latent_size) latent sample, corresponding to a traversal
    of a latent variable indicated by idx.

    Parameters
    ----------
    idx : int
        Index of continuous dimension to traverse. If the continuous latent
        vector is 10 dimensional and idx = 7, then the 7th dimension
        will be traversed while all others are fixed.

    n_samples : int
        Number of samples to generate.

    data : torch.Tensor or None, optional
        Data to use for computing the posterior. If `None` 
        then use the mean of the prior (all zeros) for all other dimensions.
    """
    model.eval()
    if data is None:
        # mean of prior for other dimensions
        samples = torch.zeros(n_samples, n_latents)
        traversals = torch.linspace(-2, 2, steps=n_samples)

    else:
        if data.size(0) > 1:
            raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

        with torch.no_grad():
            post_mean, post_logvar = model.encoder(data.to(device))
            samples = model.reparameterize(post_mean, post_logvar)
            samples = samples.cpu().repeat(n_samples, 1)
            post_mean_idx = post_mean.cpu()[0, idx]
            post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]         

        # travers from the gaussian of the posterior in case quantile
#         traversals = torch.linspace(post_mean_idx - max_traversal*post_std_idx, 
#                                     post_mean_idx + max_traversal*post_std_idx, 
#                                     steps=n_samples)
        traversals = torch.linspace(post_mean_idx - max_traversal, 
                                    post_mean_idx + max_traversal, 
                                    steps=n_samples)

    for i in range(n_samples):
        samples[i, idx] = traversals[i]

    return samples
    
    
def traversals(model,
               data=None,
               n_samples=100,
               n_latents=2,
               max_traversal=1.):
    """
    """
    latent_samples = [traverse_line(dim, model, n_samples, n_latents, data=data, max_traversal=max_traversal) for dim in range(n_latents)]
    decoded_traversal = model.decoder(torch.cat(latent_samples, dim=0).to(device))
    decoded_traversal = decoded_traversal.detach().cpu()

    return decoded_traversal


def plot_traversals(model, 
                    data, 
                    lb=0,
                    ub=2000,
                    num=100,
                    draw_data=False,
                    draw_recon=False,
                    traversal_samples=100, 
                    n_latents=4,
                    max_traversal=1.):
    if draw_data is True:
        plot_2d_samples(data[:,:2], color='C0')
    if draw_recon is True:
        recon_data, _, _ = model(data)
        plot_2d_samples(recon_data[:,:2], color='C8')    
    ranges = np.arange(lb, ub)
    samples_index = np.random.choice(ranges, num, replace=False)
    for i in samples_index:
        decoded_traversal = traversals(model, data=data[i:i+1], n_samples=traversal_samples, n_latents=n_latents,
                                       max_traversal=max_traversal)
        decoded_traversal0 = decoded_traversal[:,:2]
        plot_2d_samples(decoded_traversal0[:100], color='C2')
        plot_2d_samples(decoded_traversal0[100:200], color='C3')
        plot_2d_samples(decoded_traversal0[200:300], color='C4')
        plot_2d_samples(decoded_traversal0[300:400], color='C5')


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 2] = 1 / N
    W.view(-1)[1::M + 2] = strat_weight
    W[M, 0] = strat_weight
    return W.log()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)