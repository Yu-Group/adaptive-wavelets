import numpy as np
import matplotlib.pyplot as plt

def viz_basis(D, save=False, titles=None):
    R, C = 4,8
    i = 0
    vmin = np.min(D)
    vmax = np.max(D)
    plt.figure(figsize=(C * 3, R * 3), dpi=200)
    for r in range(R):
        for c in range(C):
            if i >= D.shape[0]:
                break
            plt.subplot(R, C, i + 1)
            plt.imshow(D[i], vmin=vmin, vmax=vmax, cmap='gray')
            plt.axis('off')
            col = 'black' if save else 'white'
            if titles is not None:
                plt.title(str(titles[i]), fontsize=25, color=col)
            i += 1
    plt.tight_layout()
    if save:
        plt.savefig('fig_basis.pdf', facecolor='w')
    else:
        plt.show()
        
def plot_scores_across_bases(results):
    
    print('shapes', preds_all.shape, scores_all.shape)
    class_num = 0
    R, C = 6, 5
    plt.figure(dpi=200)
    basis_num = 0
    for r in range(R):
        for c in range(C):
            plt.subplot(R, C, basis_num + 1)
    #         plt.plot(scores_all[:, basis_num, class_num]) / preds_all[:, 0, class_num])
            plt.plot(np.divide(scores_all[:, basis_num, class_num], preds_all[:, 0, class_num]))
            basis_num += 1
    #         plt.xaxis('off')

    plt.tight_layout()
    plt.show()