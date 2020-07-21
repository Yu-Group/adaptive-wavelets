import numpy as np
import torch
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_2d_samples(sample):
    sample_np = sample.numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y)
    
    
def traverse_line(idx, model, n_samples=100, n_latents=2, data=None):
    """
    """
    if data is None:
        # mean of prior for other dimensions
        samples = torch.zeros(n_samples, n_latents)
        traversals = torch.linspace(-1, 1, steps=n_samples)

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
        traversals = torch.linspace(post_mean-1, post_mean+1, steps=n_samples)

    for i in range(n_samples):
        samples[i, idx] = traversals[i]

    return samples
    
    
def traversals(model,
               data=None,
               n_samples=100,
               n_latents=2):
    """
    """
    latent_samples = [traverse_line(dim, model, n_samples, n_latents, data=data) for dim in range(n_latents)]
    decoded_traversal = model.decoder(torch.cat(latent_samples, dim=0).to(device))
    decoded_traversal = decoded_traversal.detach().cpu()

    return decoded_traversal