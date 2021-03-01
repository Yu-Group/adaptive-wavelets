import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import seaborn as sns
import sys
import matplotlib.pyplot as plt
from skimage.transform import rescale
sys.path.append('../..')
from os.path import join as opj
from matplotlib import gridspec
cb = '#66ccff'
cr = '#cc0000'
cm = sns.diverging_palette(10, 240, n=1000, as_cmap=True)

    
    
def cshow(im):
    plt.imshow(im, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.axis('off')
    
        
# def viz_filters(tensors, n_row=4, n_col=8, resize_fac=2, normalize=True, vmax=None, vmin=None, title=None):
#     plt.figure(figsize=(10,10))
#     # plot filters
#     p = tensors.shape[2] + 2
#     mosaic = np.zeros((p*n_row,p*n_col))
#     indx = 0
#     for i in range(n_row):
#         for j in range(n_col):
#             im = tensors.data.cpu().numpy()[indx].squeeze()
#             if normalize:
#                 im = (im-np.min(im))
#                 im = im/np.max(im)
#             mosaic[i*p:(i+1)*p,j*p:(j+1)*p] = np.pad(im,(1,1),mode='constant')
#             indx += 1
#     if title is not None:
#         plt.title(title)
#     plt.imshow(rescale(mosaic,resize_fac,mode='constant'), cmap='gray')
#     plt.axis('off')    
#     plt.show() 
    

def plot_2dreconstruct(im, recon):
    if 'Tensor' in str(type(im)):
        im = im.detach().data.cpu()
        recon = recon.detach().data.cpu()
    res = im - recon
    pl = [im, recon, res]
    
    R = 3
    C = min(im.size(0),10)
    plt.figure(figsize=(C+1, R+1), dpi=200)
    gs = gridspec.GridSpec(R, C,
             wspace=0.0, hspace=0.0, 
             top=1.-0.5/(R+1), bottom=0.5/(R+1), 
             left=0.5/(C+1), right=1-0.5/(C+1))     
    
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r,c])
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
    
    
def plot_2dfilts(filts: list, scale=2, share_min_max=True, figsize=(10, 10)):
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
        
    plt.figure(figsize=figsize, dpi=200)
    for i in range(ls):
        plt.subplot(1, ls, i + 1)
        if share_min_max:
            plt.imshow(rescale(filts[i], scale, mode='constant'), cmap='gray', extent=[0,2,0,2], vmin=v_min, vmax=v_max)
        else:
            plt.imshow(rescale(filts[i], scale, mode='constant'), cmap='gray', extent=[0,2,0,2])
        plt.axis('off')  
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
    C = min(data.size(0),10)
    plt.figure(figsize=(C+1, R+1), dpi=200)
    gs = gridspec.GridSpec(R, C,
             wspace=0.0, hspace=0.0, 
             top=1.-0.5/(R+1), bottom=0.5/(R+1), 
             left=0.5/(C+1), right=1-0.5/(C+1))     
    
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r,c])
            ax.plot(pl[r][c][0])
            ax.set_ylim((vmin-1, vmax))
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
    
    
def plot_1dfilts(filts: list, figsize=(10, 10)):
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
    titles = ['low pass', 'high pass']
        
    plt.figure(figsize=figsize, dpi=200)
    for i in range(ls):
        plt.subplot(1, ls, i + 1)
        plt.plot(filts[i])
        plt.ylim((v_min - 1, v_max + 1))
        plt.axis('off')
    plt.show()        
    
    