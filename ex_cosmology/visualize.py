import numpy as np
import matplotlib.pyplot as plt
from style import *


def visualize(im_orig, transform):
    plt.figure(dpi=100, figsize=(9, 6))
    R, C = 2, 3
    tits = ['orig', 'transformed', 'diff']
    ims = [im_orig, transform(im_orig), im_orig - transform(im_orig)]
    for i in range(3):    
        plt.subplot(R, C, i + 1)
        plt.title(tits[i])
        plt.imshow(ims[i])
        plt.axis('off')
    
    for i in range(3):
        plt.subplot(R, C, 4 + i)
        plt.imshow(np.fft.fftshift(np.abs(np.fft.fft2(ims[i]))))
        plt.xticks([0, im_orig.shape[0] / 2, im_orig.shape[0] - 1], labels=[-1, 0, 1])
        plt.yticks([0, im_orig.shape[1] / 2, im_orig.shape[1] - 1], labels=[-1, 0, 1])
        plt.xlabel('frequency x')
        plt.ylabel('frequency y')

    plt.tight_layout()
    
    
def cshow(im):
    plt.imshow(im, cmap='magma', vmax=0.15, vmin=-0.05)
    plt.axis('off')
    
    
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