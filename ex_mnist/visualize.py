import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt

def viz_basis(D, R=5, C=6, titles=None):
    i = 0
    vmin = np.min(D)
    vmax = np.max(D)
    plt.figure(figsize=(C * 3, R * 3), dpi=300)
    for r in range(R):
        for c in range(C):
            plt.subplot(R, C, i + 1)
            if titles is not None:
                plt.title(titles[i])
            plt.imshow(D[i], vmin=vmin, vmax=vmax, cmap='gray')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            i += 1
    plt.tight_layout()
    plt.show()

def viz_tensors(tensors, n_row=4, n_col=8, normalize=True, vmax=None, vmin=None):
    plt.figure(figsize=(15,15))
    # plot filters
    p = tensors.shape[2] + 2
    mosaic = np.zeros((p*n_row,p*n_col))
    indx = 0
    for i in range(n_row):
        for j in range(n_col):
            im = tensors.data.cpu().squeeze().numpy()[indx]
            if normalize:
                im = (im-np.min(im))
                im = im/np.max(im)
            mosaic[i*p:(i+1)*p,j*p:(j+1)*p] = np.pad(im,(1,1),mode='constant')
            indx += 1
    plt.title("Filters")
    plt.imshow(rescale(mosaic,4,mode='constant'), cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')    
    plt.show()  

def viz_interp_scores(list_of_x, interp_modules, results, basis_indx=0):
    num_modules = len(interp_modules)
    i = 0
    plt.figure(figsize=(num_modules * 3, num_modules * 1.5))
    for c in range(num_modules):
        interp_scores = results[interp_modules[i]]
        plt.subplot(2, num_modules, i + 1)
        plt.plot(list_of_x, interp_scores.mean(axis=0), alpha=0.5, color='blue', linewidth=4.0)
        plt.fill_between(list_of_x, interp_scores.mean(axis=0)-interp_scores.std(axis=0),
                    interp_scores.mean(axis=0)+interp_scores.std(axis=0), color='#888888', alpha=0.4)
        if 'list' not in str(type(basis_indx)):
            basis_indx = [basis_indx]
        for j in basis_indx:
            plt.axvline(x=j, linestyle='--', color='green', label='true basis{}'.format(j), linewidth=2.0)
        plt.legend()
        plt.xlabel('basis index')
        plt.ylabel('interp score')
        plt.title(interp_modules[i])

        plt.subplot(2, num_modules, i + 1 + num_modules)
        plt.hist(np.argmax(interp_scores,axis=1), bins=list_of_x-0.5, alpha=0.4)
        if 'list' not in str(type(basis_indx)):
            basis_indx = [basis_indx]
        for j in basis_indx:
            plt.axvline(x=j, linestyle='--', color='green', label='true basis{}'.format(j), linewidth=2.0)
        plt.legend()
        plt.xlabel('basis index')
        plt.ylabel('frequency')
        plt.title('Max basis index')
        i += 1
    plt.tight_layout()
    plt.show()

    
