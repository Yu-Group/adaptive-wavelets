import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
import sys
sys.path.append('../util')
from style import *


def viz_filters(model):
    n_row = 2
    n_col = 5
    plt.figure(figsize=(15,15))
    # plot filters
    mod = model.convt1
    p = mod.kernel_size[0] + 2
    mosaic = np.zeros((p*n_row,p*n_col))
    indx = 0
    for i in range(n_row):
        for j in range(n_col):
            im = mod.weight.data.cpu().squeeze().numpy()[indx]
            im = (im-np.min(im))
            im = im/np.max(im)
            mosaic[i*p:(i+1)*p,j*p:(j+1)*p] = np.pad(im,(1,1),mode='constant')
            indx += 1
    plt.title("Filters")
    plt.imshow(rescale(mosaic,4,mode='constant'), cmap='gray')
    plt.axis('off')    
    plt.show()        
    
    
def viz_im_r(im, im_r):
    im = im.data.squeeze().cpu()
    im_r = im_r.data.squeeze().cpu()

    plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray', vmax=1.0, vmin=-1.0)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_r, cmap='gray', vmax=1.0, vmin=-1.0)
    plt.title('Reconstructed')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(im-im_r, cmap='gray')
    plt.title('Difference')
    plt.axis('off')
    plt.show()    
    
    
def viz_im_a(im, im_a):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmax=1.0, vmin=0.0)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(im_a, cmap='gray', vmax=1.0, vmin=0.0)
    plt.title('Adversarial')
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
        