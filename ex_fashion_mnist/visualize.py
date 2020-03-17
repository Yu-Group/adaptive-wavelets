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
    
    
def viz_ims(im, im_r):
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