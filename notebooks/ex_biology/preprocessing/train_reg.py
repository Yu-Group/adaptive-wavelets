import os
import pickle as pkl
from os.path import join as oj

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels import robust
import features
import data
import config
from tqdm import tqdm
from scipy.stats import pearsonr, kendalltau
from neural_networks import neural_net_sklearn

#cell_nums_train = np.array([1, 2, 3, 4, 5])
#cell_nums_test = np.array([6])


def add_robust_features(df):
    df['X_95_quantile'] = np.array([np.quantile(df.iloc[i].X, 0.95) for i in range(len(df))])
    df['X_mad'] = np.array([robust.mad(df.iloc[i].X) for i in range(len(df))])
    return df


def log_transforms(df):
    
    df = add_robust_features(df)
    df['X_max_log'] = np.log(df['X_max'])
    df['X_95_quantile_log'] = np.log(df['X_95_quantile'] + 1)
    df['Y_max_log'] = np.log(df['Y_max'])
    df['X_mad_log'] = np.log(df['X_mad'])

    def calc_rise_log(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        rise = np.log(val_max) - np.log(abs(np.min(x[:idx_max + 1])) + 1)  # max change before peak
        return rise

    def calc_fall_log(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        fall = np.log(val_max) - np.log(abs(np.min(x[idx_max:])) + 1)  # drop after peak
        return fall

    df['rise_log'] = np.array([calc_rise_log(df.iloc[i].X) for i in range(len(df))])
    df['fall_log'] = np.array([calc_fall_log(df.iloc[i].X) for i in range(len(df))])
    num = 3
    df['rise_local_3_log'] = df.apply(lambda row:
                                      calc_rise_log(np.array(row['X'][max(0, row['X_peak_idx'] - num):
                                                                      row['X_peak_idx'] + num + 1])),
                                      axis=1)
    df['fall_local_3_log'] = df.apply(lambda row:
                                      calc_fall_log(np.array(row['X'][max(0, row['X_peak_idx'] - num):
                                                                      row['X_peak_idx'] + num + 1])),
                                      axis=1)

    num2 = 11
    df['rise_local_11_log'] = df.apply(lambda row:
                                       calc_rise_log(np.array(row['X'][max(0, row['X_peak_idx'] - num2):
                                                                       row['X_peak_idx'] + num2 + 1])),
                                       axis=1)
    df['fall_local_11_log'] = df.apply(lambda row:
                                       calc_fall_log(np.array(row['X'][max(0, row['X_peak_idx'] - num2):
                                                                       row['X_peak_idx'] + num2 + 1])),
                                       axis=1)
    return df

def add_sig_mean(df, resp_tracks=['Y']):
    """
    add response of regression problem: mean auxilin strength among significant observations
    """
    for track in resp_tracks:
        sig_mean = []
        for i in range(len(df)):
            r = df.iloc[i]
            sigs = np.array(r[f'{track}_pvals']) < 0.05
            if sum(sigs)>0:
                sig_mean.append(np.mean(np.array(r[track])[sigs]))
            else:
                sig_mean.append(0)
        df[f'{track}_sig_mean'] = sig_mean
        df[f'{track}_sig_mean_normalized'] = sig_mean
        for cell in set(df['cell_num']):
            cell_idx = np.where(df['cell_num'].values == cell)[0]
            y = df[f'{track}_sig_mean'].values[cell_idx]
            df[f'{track}_sig_mean_normalized'].values[cell_idx] = (y - np.mean(y))/np.std(y)
    return df


def train_reg(df, 
              feat_names, 
              model_type='rf', 
              outcome_def='Y_max_log',
              out_name='results/regression/test.pkl', 
              seed=42,
              **kwargs):
    '''
    train regression model
    
    hyperparameters of model can be specified using **kwargs
    '''
    np.random.seed(seed)
    X = df[feat_names]
    # X = (X - X.mean()) / X.std() # normalize the data
    y = df[outcome_def].values

    if model_type == 'rf':
        m = RandomForestRegressor(n_estimators=100)
    elif model_type == 'dt':
        m = DecisionTreeRegressor()
    elif model_type == 'linear':
        m = LinearRegression()
    elif model_type == 'ridge':
        m = RidgeCV()
    elif model_type == 'svm':
        m = SVR(gamma='scale')
    elif model_type == 'gb':
        m = GradientBoostingRegressor()
    elif model_type == 'irf':
        m = irf.ensemble.wrf()
    elif 'nn' in model_type: # neural nets
        
        """
        train fully connected neural network
        """
        
        H = kwargs['fcnn_hidden_neurons'] if 'fcnn_hidden_neurons' in kwargs else 40
        epochs = kwargs['fcnn_epochs'] if 'fcnn_epochs' in kwargs else 1000
        batch_size = kwargs['fcnn_batch_size'] if 'fcnn_batch_size' in kwargs else 1000
        track_name = kwargs['track_name'] if 'track_name' in kwargs else 'X_same_length'
        D_in = len(df[track_name].iloc[0])
        
        m = neural_net_sklearn(D_in=D_in, 
                             H=H, 
                             p=len(feat_names)-1,
                             epochs=epochs,
                             batch_size=batch_size,
                             track_name=track_name,
                             arch=model_type)

    # scores_cv = {s: [] for s in scorers.keys()}
    # scores_test = {s: [] for s in scorers.keys()}
    imps = {'model': [], 'imps': []}

    cell_nums_train = np.array(list(set(df.cell_num.values)))
    kf = KFold(n_splits=len(cell_nums_train))

    # split testing data based on cell num
    #idxs_test = df.cell_num.isin(cell_nums_test)
    #idxs_train = df.cell_num.isin(cell_nums_train)
    #X_test, Y_test = X[idxs_test], y[idxs_test]
    num_pts_by_fold_cv = []
    y_preds = {}
    cv_score = []
    cv_pearsonr = []
    
    print("Looping over cv...")
    # loops over cv, where test set order is cell_nums_train[0], ..., cell_nums_train[-1]
    for cv_idx, cv_val_idx in tqdm(kf.split(cell_nums_train)):
        # get sample indices
        
        
        idxs_cv = df.cell_num.isin(cell_nums_train[np.array(cv_idx)])
        idxs_val_cv = df.cell_num.isin(cell_nums_train[np.array(cv_val_idx)])
        X_train_cv, Y_train_cv = X[idxs_cv], y[idxs_cv]
        X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]
        num_pts_by_fold_cv.append(X_val_cv.shape[0])

        # resample training data

        # fit
        m.fit(X_train_cv, Y_train_cv)

        # get preds
        preds = m.predict(X_val_cv)
        y_preds[cell_nums_train[np.array(cv_val_idx)][0]] = preds
        if 'log' in outcome_def:
            cv_score.append(r2_score(np.exp(Y_val_cv), np.exp(preds)))
            cv_pearsonr.append(pearsonr(np.exp(Y_val_cv), np.exp(preds))[0])
        else:
            print(r2_score(Y_val_cv, preds))
            cv_score.append(r2_score(Y_val_cv, preds))
            cv_pearsonr.append(pearsonr(Y_val_cv, preds)[0])
    
    print("Training with full data...")
    # cv_score = cv_score/len(cell_nums_train)
    m.fit(X, y)
    #print(cv_score)
    #test_preds = m.predict(X_test)
    results = {'y_preds': y_preds,
               'y': y,
               'model_state_dict': m.model.state_dict(),
               #'test_preds': test_preds,
               'cv': {'r2': cv_score, 'pearsonr': cv_pearsonr},
               'model_type': model_type,
               #'model': m,
               'num_pts_by_fold_cv': np.array(num_pts_by_fold_cv),
               }
    if model_type in ['rf', 'linear', 'ridge', 'gb', 'svm', 'irf']:
        results['model'] = m
    # save results
    # os.makedirs(out_dir, exist_ok=True)

    pkl.dump(results, open(out_name, 'wb'))


def load_results(out_dir, by_cell=True):
    r = []
    for fname in os.listdir(out_dir):
        if os.path.isdir(oj(out_dir, fname)): 
            continue
        d = pkl.load(open(oj(out_dir, fname), 'rb'))
        metrics = {k: d['cv'][k] for k in d['cv'].keys() if not 'curve' in k}
        num_pts_by_fold_cv = d['num_pts_by_fold_cv']
        print(metrics)
        out = {k: np.average(metrics[k], weights=num_pts_by_fold_cv) for k in metrics}
        if by_cell:
            out.update({'cv_accuracy_by_cell': metrics['r2']})
        out.update({k + '_std': np.std(metrics[k]) for k in metrics})
        out['model_type'] = fname.replace('.pkl', '')  # d['model_type']
        print(d['cv'].keys())
        # imp_mat = np.array(d['imps']['imps'])
        # imp_mu = imp_mat.mean(axis=0)
        # imp_sd = imp_mat.std(axis=0)

        # feat_names = d['feat_names_selected']
        # out.update({feat_names[i] + '_f': imp_mu[i] for i in range(len(feat_names))})
        # out.update({feat_names[i]+'_std_f': imp_sd[i] for i in range(len(feat_names))})
        r.append(pd.Series(out))
    r = pd.concat(r, axis=1, sort=False).T.infer_objects()
    r = r.reindex(sorted(r.columns), axis=1)  # sort the column names
    r = r.round(3)
    r = r.set_index('model_type')
    return r

def load_and_train(dset, outcome_def, out_dir, feat_names=None, use_processed=True):
    
    df = pd.read_pickle(f'../data/tracks/tracks_{dset}.pkl')
    if dset == 'clath_aux_dynamin':
        df = df[df.catIdx.isin([1, 2])]
        df = df[df.lifetime > 15]
    else:
        df = df[df['valid'] == 1] 
    df = features.add_basic_features(df)
    df = log_transforms(df)
    df = add_sig_mean(df)
    df_train = df[df.cell_num.isin(config.DSETS[dset]['train'])] 
    df_test = df[df.cell_num.isin(config.DSETS[dset]['test'])] 
    df_train = df_train.dropna()
    
    #outcome_def = 'Z_sig_mean'
    #out_dir = 'results/regression/Sep15'
    os.makedirs(out_dir, exist_ok=True)
    if not feat_names:
        feat_names = data.get_feature_names(df_train)
        feat_names = [x for x in feat_names 
                      if not x.startswith('sc_') 
                      and not x.startswith('nmf_')
                      and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                   'X_max_around_Y_peak', 'X_max_after_Y_peak']
                      and not x.startswith('pc_')
                      and not 'log' in x
                      and not 'binary' in x
        #               and not 'slope' in x
                     ]
    for model_type in tqdm(['linear', 'gb', 'rf', 'svm', 'ridge']):
        out_name = f'{model_type}'
                        #print(out_name)
        if use_processed and os.path.exists(f'{out_dir}/{out_name}.pkl'):
            continue
        train_reg(df_train, feat_names=feat_names, model_type=model_type, 
                     outcome_def=outcome_def,
                     out_name=f'{out_dir}/{out_name}.pkl')    
        
def test_reg(df, 
             model, 
             feat_names=None, 
             outcome_def='Y_max_log',
             out_name='results/regression/test.pkl', 
             seed=42):

    np.random.seed(seed)
    if not feat_names:
        feat_names = data.get_feature_names(df)
        feat_names = [x for x in feat_names 
                      if not x.startswith('sc_') 
                      and not x.startswith('nmf_')
                      and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                   'X_max_around_Y_peak', 'X_max_after_Y_peak']
                      and not x.startswith('pc_')
                      and not 'log' in x
                      and not 'binary' in x
        #               and not 'slope' in x
                     ]
    X = df[feat_names]
    # X = (X - X.mean()) / X.std() # normalize the data
    test_preds = model.predict(X)
    results = {'preds': test_preds}
    if outcome_def in df.keys():
        y = df[outcome_def].values
        results['r2'] = r2_score(y, test_preds)
        results['pearsonr'] = pearsonr(y, test_preds)
        results['kendalltau'] = kendalltau(y, test_preds)
        
    return results
