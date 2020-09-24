import seaborn as sns
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from config import DIR_FIGS
from os.path import join as oj
cb = '#66ccff'
cr = '#cc0000'
cm = sns.diverging_palette(10, 240, n=1000, as_cmap=True)

def save_fig(fname):
    plt.savefig(oj(DIR_FIGS, fname) + '.png')