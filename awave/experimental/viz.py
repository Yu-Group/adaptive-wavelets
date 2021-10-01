import matplotlib.pyplot as plt
import numpy as np


def show_ylab(ax, ylab, fontsize_ylab):
    plt.axis('on')
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for x in ['right', 'top', 'bottom', 'left']:
        ax.spines[x].set_visible(False)
    plt.ylabel(ylab, fontsize=fontsize_ylab)


def imshow(im, annot: str = None):
    '''
    Params
    ------
    annot
        str to put in top-right corner
    '''

    # if 4d, take first image
    if len(im.shape) > 3:
        im = im[0]

    # if channels dimension first, transpose
    if im.shape[0] == 3 and len(im.shape) == 3:
        im = im.transpose()

    ax = plt.gca()
    ax.imshow(im)
    ax.axis('off')

    if annot is not None:
        padding = 5
        ax.annotate(
            s=annot,
            fontsize=12,
            xy=(0, 0),
            xytext=(padding - 1, -(padding - 1)),
            textcoords='offset pixels',
            bbox=dict(facecolor='white', alpha=1, pad=padding),
            va='top',
            ha='left')


def plot_grid(images, ylabs=[], annot_list=None, suptitle=None, emphasize_col: int = None, fontsize_ylab=25, cmap=None):
    '''
    Params
    ------
    images: np.ndarray or list
        (R, C, H, W, C) or (R * C, H, W, C)
    emphasize_col
        which column to emphasize (by not removing black border)
    '''

    # deal with inputs
    if type(images) == list:
        images = np.array(images)
        # print(images.shape)
    # check if wasn't passed a grid
    if len(images.shape) == 4:
        if ylabs is not None:
            R = len(ylabs)
            N_IMS = images.shape[0]
            C = N_IMS // R
        else:
            N_IMS = images.shape[0]
            R = int(np.sqrt(N_IMS))
            C = R + 1
    else:
        R = images.shape[0]
        C = images.shape[1]
        N_IMS = R * C
        # reshape to be (R * C, H, W, C)
        images = images.reshape((R * C, *images.shape[2:]))
    if annot_list is None:
        annot_list = [None] * N_IMS
    i = 0
    fig = plt.figure(figsize=(C * 3, R * 3))
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(R, C, i + 1)
            imshow(images[r * C + c].squeeze(), annot=annot_list[i])

            if c == 0 and len(ylabs) > r:
                show_ylab(ax, ylabs[r], fontsize_ylab=fontsize_ylab)

            i += 1
            if i >= images.shape[0]:
                break

            if c == emphasize_col:
                emphasize_box(ax)

    if suptitle is not None:
        fig.text(0.5, 1, suptitle, ha='center')

    '''
    if ylabs is not None:
        for radius in range(R):
            fig.text(0, radius / R + 0.5 / R, ylabs[R - 1 - radius], rotation=90,
                         va='center', fontsize=fontsize_ylab)
    '''
    fig.tight_layout()


def emphasize_box(ax):
    plt.axis('on')
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for x in ['right', 'top', 'bottom', 'left']:
        ax.spines[x].set_visible(True)
        ax.spines[x].set_linewidth(3)  # ['linewidth'] = 10

#         [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
#     ax.spines['top'].set_visible(True)
