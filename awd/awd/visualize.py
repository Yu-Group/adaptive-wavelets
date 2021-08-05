import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from skimage.transform import rescale


def cshow(im):
    plt.imshow(im, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.axis('off')


def plot_2dreconstruct(im, recon):
    if 'Tensor' in str(type(im)):
        im = im.detach().data.cpu()
        recon = recon.detach().data.cpu()
    res = im - recon
    pl = [im, recon, res]

    R = 3
    C = min(im.size(0), 10)
    plt.figure(figsize=(C + 1, R + 1), dpi=200)
    gs = gridspec.GridSpec(R, C,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (R + 1), bottom=0.5 / (R + 1),
                           left=0.5 / (C + 1), right=1 - 0.5 / (C + 1))

    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r, c])
            ax.imshow(pl[r][c][0], cmap='magma', vmax=0.15, vmin=-0.05)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
    plt.show()


def plot_2dfilts(filts: list, scale=2, share_min_max=True, figsize=(1, 1)):
    '''Plot filters in the list
    Params
    ------
    filts: list
        list of filters
    figsize: tuple
        figure size    
    '''
    ls = len(filts)
    v_min = 1e4
    v_max = -1e4
    for i in range(ls):
        v_min = min(filts[i].min(), v_min)
        v_max = max(filts[i].max(), v_max)

    fig = plt.figure(figsize=figsize, dpi=200)
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    i = 0
    for r in range(2):
        for c in range(2):
            ax = plt.subplot(gs[r, c])
            if share_min_max:
                ax.imshow(rescale(filts[i], scale, mode='constant'), cmap='gray', vmin=v_min, vmax=v_max)
            else:
                ax.imshow(rescale(filts[i], scale, mode='constant'), cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
            i += 1
    plt.tight_layout()
    plt.show()


def plot_1dreconstruct(data, recon):
    if 'Tensor' in str(type(data)):
        data = data.detach().data.cpu()
        recon = recon.detach().data.cpu()
    res = data - recon
    pl = [data, recon, res]
    vmax = torch.max(data).item()
    vmin = torch.min(data).item()

    R = 3
    C = min(data.size(0), 10)
    plt.figure(figsize=(C + 1, R + 1), dpi=200)
    gs = gridspec.GridSpec(R, C,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (R + 1), bottom=0.5 / (R + 1),
                           left=0.5 / (C + 1), right=1 - 0.5 / (C + 1))

    labs = ['Original', 'Reconstruction', 'Residual']
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r, c])
            ax.plot(pl[r][c][0])
            ax.set_ylim((vmin - 1, vmax))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
            if c == 0:
                plt.ylabel(labs[r])
    plt.show()


def plot_1dfilts(filts: list, is_title=False, figsize=(10, 10)):
    '''Plot filters in the list
    Params
    ------
    filts: list
        list of filters
    figsize: tuple
        figure size    
    '''
    ls = len(filts)
    v_min = 1e4
    v_max = -1e4

    for i in range(ls):
        v_min = min(filts[i].min(), v_min)
        v_max = max(filts[i].max(), v_max)
    titles = ['lowpass', 'highpass']

    plt.figure(figsize=figsize, dpi=200)
    for i in range(ls):
        plt.subplot(1, ls, i + 1)
        plt.plot(filts[i])
        plt.ylim((v_min - 1, v_max + 1))
        plt.axis('off')
        if is_title is True:
            plt.title(titles[i])
    plt.show()


def plot_wavefun(waves: tuple, is_title=False, figsize=(10, 10), flip_wavelet=False):
    '''Plot filters in the list
    Params
    ------
    waves: tuple
        tuple of scaling and wavelet functions
    figsize: tuple
        figure size    
    '''

    titles = ['scaling', 'wavelet']
    plt.figure(figsize=figsize, dpi=300)
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(waves[-1], waves[i])
        plt.axis('off')
        if is_title is True:
            plt.title(titles[i])
    plt.show()
