import sys
sys.path.append('../../hierarchical-dnn-interpretations') # if pip install doesn't work
import acd
from acd.scores import cd_propagate
import numpy as np
import seaborn as sns
import matplotlib.colors
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap


def calc_cd_score(xtrack_t, xfeats_t, start, stop, model):
    with torch.no_grad():
        rel, irrel = cd_propagate.propagate_lstm(xtrack_t.unsqueeze(-1), model.lstm, start=start, stop=stop, my_device='cpu')
    rel = rel.squeeze(1)
    irrel = irrel.squeeze(1)
    rel, irrel = cd_propagate.propagate_conv_linear(rel, irrel, model.fc)
    #return rel.item()
    return rel.data.numpy()

def plot_segs(track_segs, cd_scores, xtrack,
              pred=None, y=None, vabs=None, cbar=True, xticks=True, yticks=True):
    '''Plot a single segmentation plot
    '''
#     cm = sns.diverging_palette(22, 220, as_cmap=True, center='light')
    cm = LinearSegmentedColormap.from_list(
        name='orange-blue', 
        colors=[(222/255, 85/255, 51/255),'lightgray', (50/255, 129/255, 168/255)]
    )
    if vabs is None:
        vabs = np.max(np.abs(cd_scores))
    norm = matplotlib.colors.Normalize(vmin=-vabs, vmax=vabs)
    #vabs = 1.2
    # plt.plot(xtrack, zorder=0, lw=2, color='#111111')
    for i in range(len(track_segs)):
        (s, e) = track_segs[i]
        cd_score = cd_scores[i]
        seq_len = e - s
        xs = np.arange(s, e)
        if seq_len > 1:
            cd_score = [cd_score] * seq_len
            col = cm(norm(cd_score[0]))
            while len(col) == 1:
                col = col[0]
            plt.plot(xs, xtrack[s: e], zorder=0, lw=2, color=col, alpha=0.5)
        plt.scatter(xs, xtrack[s: e],
                    c=cd_score, cmap=cm, vmin=-vabs, vmax=vabs, s=6)
    if pred is not None:
        plt.title(f"Pred: {pred: .1f}, y: {y}", fontsize=24)
    if cbar:
        cb = plt.colorbar() #label='CD Score')
        cb.outline.set_visible(False)
    if not xticks:
        plt.xticks([])
    if not yticks:
        plt.yticks([])
    return cb
    
    
    
def max_abs_sum_seg(scores_list, min_length: int=1):
    """
    score_list[i][j] is the score for the segment from i to j (inclusive)
    Params
    ------
    min_length
        Minimum allowable length for a segment
    """
    
    n = len(scores_list[0])
    res = [0]*n
    paths = {}
    for s in range(n):
        for e in range(s, n):
            if e - s >= min_length - 1:
                scores_list[s][e] = abs(scores_list[s][e])
            else:
                scores_list[s][e] = -10000
    paths[-1] = []
    res[0] = scores_list[0][0]
    paths[0] = [0]
    for i in (range(1, n)):
        cand = [res[j-1] + scores_list[j][i] for j in range(i + 1)]
        seg_start = np.argmax(cand)
        res[i] = max(cand)
        paths[i] = paths[seg_start - 1] + [seg_start]
    return res, paths