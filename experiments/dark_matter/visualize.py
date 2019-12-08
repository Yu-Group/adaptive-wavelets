import numpy as np
import matplotlib.pyplot as plt

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