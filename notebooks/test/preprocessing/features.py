import os
from copy import deepcopy
from os.path import join as oj

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
#from viz import *
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn import decomposition
import trend_filtering
import data
from scipy.stats import skew, pearsonr



def add_pcs(df):
    '''adds 10 pcs based on feature names
    '''
    feat_names = data.get_feature_names(df)
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    pca = decomposition.PCA(whiten=True)
    pca.fit(X[df.valid])
    X_reduced = pca.transform(X)
    for i in range(10):
        df['pc_' + str(i)] = X_reduced[:, i]
    return df


def add_dict_features(df, sc_comps_file='processed/dictionaries/sc_12_alpha=1.pkl',
                      nmf_comps_file='processed/dictionaries/nmf_12.pkl',
                      use_processed=True):
    '''Add features from saved dictionary to df
    '''

    def sparse_code(X_mat, n_comps=12, alpha=1, out_dir='processed/dictionaries'):
        print('sparse coding...')
        d = DictionaryLearning(n_components=n_comps, alpha=alpha, random_state=42)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'sc_{n_comps}_alpha={alpha}.pkl'), 'wb'))

    def nmf(X_mat, n_comps=12, out_dir='processed/dictionaries'):
        print('running nmf...')
        d = NMF(n_components=n_comps, random_state=42)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'nmf_{n_comps}.pkl'), 'wb'))

    X_mat = extract_X_mat(df)
    X_mat -= np.min(X_mat)

    # if feats don't exist, compute them
    if not use_processed or not os.path.exists(sc_comps_file):
        os.makedirs('processed/dictionaries', exist_ok=True)
        sparse_code(X_mat)
        nmf(X_mat)

    try:
        # sc
        d_sc = pkl.load(open(sc_comps_file, 'rb'))
        encoding = d_sc.transform(X_mat)
        for i in range(encoding.shape[1]):
            df[f'sc_{i}'] = encoding[:, i]

        # nmf
        d_nmf = pkl.load(open(nmf_comps_file, 'rb'))
        encoding_nmf = d_nmf.transform(X_mat)
        for i in range(encoding_nmf.shape[1]):
            df[f'nmf_{i}'] = encoding_nmf[:, i]
    except:
        print('dict features not added!')
    return df


def add_smoothed_splines(df,
                         method='spline',
                         s_spl=0.004):
    X_smooth_spl = []
    X_smooth_spl_dx = []
    X_smooth_spl_d2x = []

    def num_local_maxima(x):
        return (len([i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] > x[i + 1]]))

    for x in df['X']:
        spl = UnivariateSpline(x=range(len(x)),
                               y=x,
                               w=[1.0 / len(x)] * len(x),
                               s=np.var(x) * s_spl)
        spl_dx = spl.derivative()
        spl_d2x = spl_dx.derivative()
        X_smooth_spl.append(spl(range(len(x))))
        X_smooth_spl_dx.append(spl_dx(range(len(x))))
        X_smooth_spl_d2x.append(spl_d2x(range(len(x))))
    df['X_smooth_spl'] = np.array(X_smooth_spl)
    df['X_smooth_spl_dx'] = np.array(X_smooth_spl_dx)
    df['X_smooth_spl_d2x'] = np.array(X_smooth_spl_d2x)
    df['X_max_spl'] = np.array([np.max(x) for x in X_smooth_spl])
    df['dx_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_dx])
    df['d2x_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_d2x])
    df['num_local_max_spl'] = np.array([num_local_maxima(x) for x in X_smooth_spl])
    df['num_local_min_spl'] = np.array([num_local_maxima(-1 * x) for x in X_smooth_spl])

    # linear fits
    x = np.arange(5).reshape(-1, 1)
    df['end_linear_fit'] = [LinearRegression().fit(x, end).coef_[0] for end in df['X_ends']]
    df['start_linear_fit'] = [LinearRegression().fit(x, start).coef_[0] for start in df['X_starts']]
    return df


def add_trend_filtering(df):
    df_tf = deepcopy(df)
    for i in range(len(df)):
        df_tf['X'].iloc[i] = trend_filtering.trend_filtering(y=df['X'].iloc[i], vlambda=len(df['X'].iloc[i]) * 5,
                                                             order=1)
    df_tf = add_features(df_tf)
    feat_names = data.get_feature_names(df_tf)
    feat_names = [x for x in feat_names
                  if not x.startswith('sc_')
                  and not x.startswith('nmf_')
                  and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                'X_max_around_Y_peak', 'X_max_after_Y_peak', 'X_max_diff_after_Y_peak',
                                'X_tf']
                  and not x.startswith('pc_')
                  #               and not 'local' in x
                  #               and not 'X_peak' in x
                  #               and not 'slope' in x
                  #               and not x in ['fall_final', 'fall_slope', 'fall_imp', 'fall']
                  ]
    for feat in feat_names:
        df[feat + '_tf_smooth'] = df_tf[feat]
    return df


def add_basic_features(df):
    '''Add a bunch of extra features to the df based on df.X, df.X_extended, df.Y, df.lifetime
    '''
    df = df[df.lifetime > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_max_extended'] = np.array([max(x) for x in df.X_extended.values])
    df['X_min'] = np.array([min(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    df['X_peak_idx'] = np.nan_to_num(np.array([np.argmax(x) for x in df.X]))
    df['Y_peak_idx'] = np.nan_to_num(np.array([np.argmax(y) for y in df.Y]))
    df['X_peak_time_frac'] = df['X_peak_idx'].values / df['lifetime'].values
#     df['slope_end'] = df.apply(lambda row: (row['X_max'] - row['X'][-1]) / (row['lifetime'] - row['X_peak_idx']),
#                                axis=1)
    df['X_peak_last_15'] = df['X_peak_time_frac'] >= 0.85
    df['X_peak_last_5'] = df['X_peak_time_frac'] >= 0.95

    # hand-engineeredd features
    def calc_rise(x):
        '''max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[:idx_max + 1])

    def calc_fall(x):
        '''max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[idx_max:])

    def calc_rise_slope(x):
        '''slope to max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_early = x[:idx_max + 1]
        idx_min = np.argmin(x_early)
        denom = (idx_max - idx_min)
        if denom == 0:
            return 0
        return (val_max - np.min(x_early)) / denom

    def calc_fall_slope(x):
        '''slope to max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_late = x[idx_max:]
        idx_min = np.argmin(x_late)
        denom = idx_min
        if denom == 0:
            return 0
        return (val_max - np.min(x_late)) / denom

    def max_diff(x):
        return np.max(np.diff(x))

    def min_diff(x):
        return np.min(np.diff(x))

    df['rise'] = df.apply(lambda row: calc_rise(row['X']), axis=1)
    df['fall'] = df.apply(lambda row: calc_fall(row['X']), axis=1)
    df['rise_extended'] = df.apply(lambda row: calc_rise(row['X_extended']), axis=1)
    df['fall_extended'] = df.apply(lambda row: calc_fall(row['X_extended']), axis=1)
    df['fall_late_extended'] = df.apply(lambda row: row['fall_extended'] if row['X_peak_last_15'] else row['fall'],
                                        axis=1)
    # df['fall_final'] = df.apply(lambda row: row['X'][-3] - row['X'][-1], axis=1)

    df['rise_slope'] = df.apply(lambda row: calc_rise_slope(row['X']), axis=1)
    df['fall_slope'] = df.apply(lambda row: calc_fall_slope(row['X']), axis=1)
    num = 3
    df['rise_local_3'] = df.apply(lambda row:
                                  calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num):
                                                              row['X_peak_idx'] + num + 1])),
                                  axis=1)
    df['fall_local_3'] = df.apply(lambda row:
                                  calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num):
                                                              row['X_peak_idx'] + num + 1])),
                                  axis=1)

    num2 = 11
    df['rise_local_11'] = df.apply(lambda row:
                                   calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num2):
                                                               row['X_peak_idx'] + num2 + 1])),
                                   axis=1)
    df['fall_local_11'] = df.apply(lambda row:
                                   calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num2):
                                                               row['X_peak_idx'] + num2 + 1])),
                                   axis=1)
    df['max_diff'] = df.apply(lambda row: max_diff(row['X']), axis=1)
    df['min_diff'] = df.apply(lambda row: min_diff(row['X']), axis=1)

    # imputed feats
    d = df[['X_max', 'X_mean', 'lifetime', 'rise', 'fall']]
    d = d[df['X_peak_time_frac'] <= 0.8]
#     m = RidgeCV().fit(d[['X_max', 'X_mean', 'lifetime', 'rise']], d['fall'])
#     fall_pred = m.predict(df[['X_max', 'X_mean', 'lifetime', 'rise']])
#     fall_imp = df['fall']
#     fall_imp[df['X_peak_time_frac'] > 0.8] = fall_pred[df['X_peak_time_frac'] > 0.8]
#     df['fall_imp'] = fall_imp

    return df

def extract_X_mat(df):
    '''Extract matrix for X filled with zeros after sequences
    Width of matrix is length of longest lifetime
    '''
    p = df.lifetime.max()
    n = df.shape[0]
    X_mat = np.zeros((n, p)).astype(np.float32)
    X = df['X'].values
    for i in range(n):
        x = X[i]
        num_timepoints = min(p, len(x))
        X_mat[i, :num_timepoints] = x[:num_timepoints]
    X_mat = np.nan_to_num(X_mat)
    X_mat -= np.min(X_mat)
    X_mat /= np.std(X_mat)
    return X_mat


def add_binary_features(df, outcome_def):
    '''binarize features at the difference between the mean of each class
    '''
    feat_names = data.get_feature_names(df)
    threshes = (df[df[outcome_def] == 1].mean() + df[df[outcome_def] == 0].mean()) / 2
    for i, k in tqdm(enumerate(feat_names)):
        thresh = threshes.loc[k]
        df[k + '_binary'] = df[k] >= thresh
    return df

def add_dasc_features(df, bins=100, by_cell=True):
    """
    add DASC features from Wang et al. 2020 paper
    
    Parameters:
        df: pd.DataFrame
        
        bins: int
            number of bins 
            default value is 100: the intensity level of clathrin is assigned to 100 equal-length bins
            from vmin(min intensity across all tracks) to vmax(max intensity across all tracks)
            
        by_cell: Boolean
            whether to do binning within each cell
    """
    x_dist = {}
    n = len(df)
    
    # gather min and max clathrin intensity within each cell
    if by_cell == True:
        for cell in set(df['cell_num']):
            x = []
            cell_idx = np.where(df['cell_num'].values == cell)[0]
            for i in cell_idx:
                x += df['X'].values[i]
            x_dist[cell] = (min(x), max(x))
    else:
        x = []
        for i in range(n):
            x += df['X'].values[i]
        for cell in set(df['cell_num']):
            x_dist[cell] = (min(x), max(x))
    
    # transform the clathrin intensity to a value between 0 to 100
    X_quantiles = []
    for i in range(n):
        r = df.iloc[i]
        cell = r['cell_num']
        X_quantiles.append([np.int(1.0*bins*(x - x_dist[cell][0])/(x_dist[cell][1] - x_dist[cell][0])) if not np.isnan(x) else 0 for x in r['X']])
    df['X_quantiles'] = X_quantiles
    
    # compute transition probability between different intensities, for different frames
    trans_prob = {}
    tmax = max([len(df['X_quantiles'].values[i]) for i in range(len(df))])
    for t in range(tmax - 1):
        int_pairs = []
        for i in range(n):
            if len(df['X_quantiles'].values[i]) > t + 1:
                int_pairs.append([df['X_quantiles'].values[i][t], df['X_quantiles'].values[i][t + 1]])
        int_pairs = np.array(int_pairs)
        trans_prob_t = {}
        for i in range(bins + 1):
            x1 = np.where(int_pairs[:,0]== i)[0]
            lower_states_num = np.zeros((i, 2))
            for j in range(len(int_pairs)):
                if int_pairs[j, 0] < i:
                    lower_states_num[int_pairs[j, 0], 0] += 1
                    if int_pairs[j, 1] == i:
                        lower_states_num[int_pairs[j, 0], 1] += 1
            lower_prob = [1.*lower_states_num[k, 1]/lower_states_num[k, 0] for k in range(i) if lower_states_num[k, 0] > 0]
            trans_prob_t[i] = (np.nanmean(int_pairs[x1,1] < i), 
                               #np.nanmean(int_pairs[x1,1] > i)
                              sum(lower_prob)
                              )  
        trans_prob[t] = trans_prob_t
        
    # compute D sequence 
    X_d = [[] for i in range(len(df))]
    for i in range(len(df)):
        for j, q in enumerate(df['X_quantiles'].values[i][:-1]):
            probs = trans_prob[j][q]
            if 0 < probs[0] and 0 < probs[1]:
                X_d[i].append(np.log(probs[0]/probs[1]))
            else:
                X_d[i].append(0)
                
    # compute features
    d1 = [np.mean(x) for x in X_d]
    d2 = [np.log(max((np.max(x) - np.min(x))/len(x), 1e-4)) for x in X_d]
    d3 = [skew(x) for x in X_d]
    df['X_d1'] = d1
    df['X_d2'] = d2
    df['X_d3'] = d3
    
    return df

def downsample(x, length, padding='end'):
    
    """
    downsample (clathrin) track
    
    Parameters:
    ==========================================================
        x: list
            original clathrin track (of different lengths)
            
        length: int
            length of track after downsampling
            
    Returns:
    ==========================================================
        x_ds: list
            downsampled track
    """
    
    x = np.array(x)[np.where(np.isnan(x) == False)]
    n = len(x)
    if n >= length:   
        # if length of original track is greater than targeted length, downsample     
        x_ds = [x[np.int(1.0 * (n-1) * i/(length - 1))] for i in range(length)]
    else:
        # if length of original track is smaller than targeted length, fill the track with 0s        
        if padding == 'front':
            x_ds = [0]*(length - len(x)) + list(x)
        else:
            x_ds = list(x) + [0]*(length - len(x))
    return x_ds

def downsample_video(x, length):
    
    """
    downsample video feature in the same way
    """
   
    n = len(x)    
    if n >= length:   
        # if length of original track is greater than targeted length, downsample 
        time_index = [np.int(1.0 * (n-1) * i/(length - 1)) for i in range(length)]
        x_ds = x[time_index, :, :]
    elif n > 0:
        # if length of original track is smaller than targeted length, fill the track with 0s
        x_ds = np.vstack((x, np.zeros((length - n, 10, 10))))
    else:
        x_ds = np.zeros((40, 10, 10))
    return x_ds

def normalize_track(df, track='X_same_length', by_time_point=True):
    
    """
    normalize tracks
    """
    
    df[f'{track}_normalized'] = df[track].values
    for cell in set(df['cell_num']):
        cell_idx = np.where(df['cell_num'].values == cell)[0]
        y = df[track].values[cell_idx]
        y = np.array(list(y))
        if by_time_point:
            df[f'{track}_normalized'].values[cell_idx] = list((y - np.mean(y, axis=0))/np.std(y, axis=0))
        else:
            df[f'{track}_normalized'].values[cell_idx] = list((y - np.mean(y))/np.std(y))
    return df

def normalize_feature(df, feat):
    
    """
    normalize scalar features
    """
    df = df.astype({feat: 'float64'})
    for cell in set(df['cell_num']):
        cell_idx = np.where(df['cell_num'].values == cell)[0]
        y = df[feat].values[cell_idx]
            #y = np.array(list(y))
        df[feat].values[cell_idx] = (y - np.mean(y))/np.std(y)
    return df

def normalize_video(df, video='X_video'):
    
    """
    normalize videos (different frames are normalized separately)
    
    e.g. to normalize the first frame, we take the first frame of all videos,
    flatten and concatenate them into one 1-d array, 
    and extract the mean and std
    """
    
    df[f'{video}_normalized'] = df[video].values
    for cell in set(df['cell_num']):
        cell_idx = np.where(df['cell_num'].values == cell)[0]
        y = df[video].values[cell_idx]
        video_shape = y[0].shape
        video_mean, video_std = np.zeros(video_shape), np.zeros(video_shape)
        for j in (range(video_shape[0])):
            all_frames_j = np.array([y[i][j].reshape(1, -1)[0] for i in range(len(y))]).reshape(1, -1)[0]
            video_mean[j] = np.mean(all_frames_j) * np.ones((video_shape[1], video_shape[2]))
            video_std[j] = np.std(all_frames_j) * np.ones((video_shape[1], video_shape[2]))
        df[f'{video}_normalized'].values[cell_idx] = list((list(y) - video_mean)/(video_std))
    return df    
    
    