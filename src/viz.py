import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import seaborn as sns
import sys
import matplotlib.pyplot as plt
from skimage.transform import rescale
sys.path.append('../..')
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
    
        
def viz_filters(tensors, n_row=4, n_col=8, resize_fac=2, normalize=True, vmax=None, vmin=None, title=None):
    plt.figure(figsize=(10,10))
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
    plt.imshow(rescale(mosaic,resize_fac,mode='constant'), cmap='gray')
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
    
    
def viz_list(x: list, figsize=(10, 10), scale=2):
    '''Plot images in the list
    Params
    ------
    x: list
        list of images
    figsize: tuple
        figure size    
    '''
    ls = len(x)
    x_min = 1e4
    x_max = -1e4
    for i in range(ls):
        x_min = min(x[i].min(), x_min)
        x_max = max(x[i].max(), x_max)
        
    plt.figure(figsize=figsize, dpi=200)
    for i in range(ls):
        plt.subplot(1, ls, i + 1)
        plt.imshow(rescale(x[i], scale, mode='constant'), cmap='gray', extent=[0,2,0,2], vmin=x_min, vmax=x_max)
        plt.axis('off')  
    plt.tight_layout()
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

    