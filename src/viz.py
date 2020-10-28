import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import seaborn as sns
import sys
import matplotlib.pyplot as plt
from skimage.transform import rescale
sys.path.append('..')
from config import DIR_FIGS
from os.path import join as opj
cb = '#66ccff'
cr = '#cc0000'
cm = sns.diverging_palette(10, 240, n=1000, as_cmap=True)

def save_fig(fname):
    plt.savefig(opj(DIR_FIGS, fname) + '.png')
    
    
def cshow(im):
    plt.imshow(im, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.axis('off')
    
    
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
        
        
def viz_filters(tensors, n_row=4, n_col=8, resize_fac=2, normalize=True, vmax=None, vmin=None, title=None):
    plt.figure(figsize=(15,15))
    # plot filters
    p = tensors.shape[2] + 2
    mosaic = np.zeros((p*n_row,p*n_col))
    indx = 0
    for i in range(n_row):
        for j in range(n_col):
            im = tensors.data.cpu().numpy()[indx].squeeze()
            if normalize:
                im = (im-np.min(im))
                im = im/np.max(im)
            mosaic[i*p:(i+1)*p,j*p:(j+1)*p] = np.pad(im,(1,1),mode='constant')
            indx += 1
    if title is not None:
        plt.title(title)
    plt.imshow(rescale(mosaic,resize_fac,mode='constant'), cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')    
    plt.show() 
    
    
def viz_im_r(im, im_r):
    if 'Tensor' in str(type(im)):
        im = im.data.squeeze().cpu()
        im_r = im_r.data.squeeze().cpu()

    plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_r, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.title('Reconstructed')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(im-im_r, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.title('Difference')
    plt.axis('off')
    plt.show()        
    
    
def plot_all(bs, scores_list, preds_list, params_list, class_num=1, param_num=None, tit=None, ylab=True):
    s = scores_list[..., class_num].T / preds_list[:, class_num] # (num_bands, num_curves)
    
    if param_num is not None:
        if tit is None:
            s = s[:, param_num::len(params_list)] # skip every 10 images
            params = params_list[param_num]
            plt.title(f"$m_\\nu=${params[0]:0.2f}   $\Omega_m$={params[1]:0.2f}    $10^9A_s$={params[2]:0.2f}")
#         plt.title(str())
        else:
            plt.title(tit)
    else:
        if tit is None:
            plt.title('all params')
        else:
            plt.title(tit)
    
#     print(bs)
    plt.plot(bs, np.array(s), '-', alpha=0.1, color=cb)
    plt.plot(bs, np.array(s).mean(axis=1), '-', color=cr)
    
    plt.xlabel('Central scale (angular multipole $\ell$)') # $\pm 1350$')
    plt.ylabel('CD Score (normalized)')     

    