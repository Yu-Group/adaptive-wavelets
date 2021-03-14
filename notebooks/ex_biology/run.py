
import os
from os.path import join as oj
import sys
sys.path.append('preprocessing')
import numpy as np
import pandas as pd
import torch
from math import floor
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle as pkl
import data
from sklearn.decomposition import DictionaryLearning, NMF


track_name = 'X_same_length_normalized'
track_name_unnormalized = 'X_same_length'
dsets = ['clath_aux+gak_a7d2']
splits = ['train', 'test']

meta = ['cell_num', 'Y_sig_mean', 'Y_sig_mean_normalized']

try:
    df = pd.read_pickle('data/df_py36.pkl')
    df_test = pd.read_pickle('data/df_test_py36.pkl')
except:
    dfs, _ = data.load_dfs_for_lstm(dsets=dsets, splits=splits, meta=meta)
    df = dfs[('clath_aux+gak_a7d2', 'train')]
    df.to_pickle('data/df_py36.pkl', protocol=3)
    df_test = dfs[('clath_aux+gak_a7d2', 'test')]
    df_test.to_pickle('data/df_test_py36.pkl', protocol=3)

# load model
results = pkl.load(open('models/dnn_full_long_normalized_across_track_1_feat.pkl', 'rb'))
dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm')
dnn.model.load_state_dict(results['model_state_dict'])

# training data

# input to the model (n x 40)
X = np.vstack([x for x in df[track_name].values])
# input before normalization, can be used for interpretation (n x 40)
X_unnormalized = np.vstack([x for x in df[track_name_unnormalized].values])
y = df['y_consec_thresh'].values

# test data
# input to the model (n x 40)
X_test = np.vstack([x for x in df_test[track_name].values])
# input before normalization, can be used for interpretation (n x 40)
X_unnormalized_test = np.vstack([x for x in df_test[track_name_unnormalized].values])
y_test = df_test['y_consec_thresh'].values