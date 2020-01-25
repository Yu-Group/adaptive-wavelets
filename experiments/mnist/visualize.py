import numpy as np
import matplotlib.pyplot as plt

def viz_basis(D, R=5, C=6):
    i = 0
    vmin = np.min(D)
    vmax = np.max(D)
    plt.figure(figsize=(C * 3, R * 3), dpi=200)
    for r in range(R):
        for c in range(C):
            plt.subplot(R, C, i + 1)
            plt.imshow(D[i], vmin=vmin, vmax=vmax, cmap='viridis')
            plt.axis('off')
            i += 1
    plt.tight_layout()
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
