import os,sys
opj = os.path.join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
sys.path.append('../../../notebooks/ex_biology/preprocessing')
import data
import neural_networks

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_dataloader(root_dir, shuffle=True, pin_memory=True, batch_size=64, is_continuous=False, **kwargs):
    """A generic data loader

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.   

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    
    track_name = 'X_same_length_normalized'
    df = pd.read_pickle(opj(root_dir, 'df_py36.pkl'))
    df_test = pd.read_pickle(opj(root_dir, 'df_test_py36.pkl'))
                             
    # training data
    # input to the model (n x 40)
    X = np.vstack([x for x in df[track_name].values])
    X = X.reshape(-1,1,40)
    y = df['y_consec_thresh'].values if is_continuous is False else df['Y_sig_mean_normalized'].values  
                             
    # test data
    # input to the model (n x 40)
    X_test = np.vstack([x for x in df_test[track_name].values])
    X_test = X_test.reshape(-1,1,40)
    y_test = df_test['y_consec_thresh'].values if is_continuous is False else df_test['Y_sig_mean_normalized'].values     
                                                          
    inputs = torch.tensor(X, dtype=torch.float)
    labels = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              shuffle=shuffle) 

    inputs_test = torch.tensor(X_test, dtype=torch.float)
    labels_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)
    dataset_test = TensorDataset(inputs_test, labels_test)
    test_loader = DataLoader(dataset_test, 
                             batch_size=batch_size,
                             shuffle=False)                              
    
    return (train_loader, test_loader)
    