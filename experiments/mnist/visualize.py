import numpy as np
import matplotlib.pyplot as plt

def viz_basis(D):
    R, C = 5,6
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