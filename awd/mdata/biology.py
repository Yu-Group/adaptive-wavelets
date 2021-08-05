import os
import sys

opj = os.path.join
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

sys.path.append('../../data/biology')
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle as pkl

from awd.models.models import LSTMNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    X = X.reshape(-1, 1, 40)
    y = df['y_consec_thresh'].values if is_continuous is False else df['Y_sig_mean_normalized'].values

    # test data
    # input to the model (n x 40)
    X_test = np.vstack([x for x in df_test[track_name].values])
    X_test = X_test.reshape(-1, 1, 40)
    y_test = df_test['y_consec_thresh'].values if is_continuous is False else df_test['Y_sig_mean_normalized'].values

    inputs = torch.tensor(X, dtype=torch.float)
    labels = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory)

    inputs_test = torch.tensor(X_test, dtype=torch.float)
    labels_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)
    dataset_test = TensorDataset(inputs_test, labels_test)
    test_loader = DataLoader(dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def load_pretrained_model(root_dir, device=device):
    """load pretrained model for interpretation
    """
    results = pkl.load(open(opj(root_dir, 'dnn_full_long_normalized_across_track_1_feat.pkl'), 'rb'))
    dnn = neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm')
    dnn.model.load_state_dict(results['model_state_dict'])
    m = deepcopy(dnn.model)
    m = m.eval()
    # freeze layers
    for param in m.parameters():
        param.requires_grad = False
    model = ReshapeModel(m)
    return model


class ReshapeModel(torch.nn.Module):
    def __init__(self, model):
        super(ReshapeModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = x.squeeze()
        return self.model(x)


class neural_net_sklearn():
    """
    sklearn wrapper for training a neural net
    """

    def __init__(self, D_in=40, H=40, p=17, epochs=1000, batch_size=100, track_name='X_same_length_normalized',
                 arch='fcnn', torch_seed=2):

        """
        Parameters:
        ==========================================================
            D_in, H, p: int
                same as input to FCNN
                
            epochs: int
                number of epochs
                
            batch_size: int
                batch size
                
            track_name: str
                column name of track (the tracks should be of the same length)
        """

        torch.manual_seed(torch_seed)
        self.D_in = D_in
        self.H = H
        self.p = p
        self.epochs = epochs
        self.batch_size = batch_size
        self.track_name = track_name
        self.torch_seed = torch_seed
        self.arch = arch

        torch.manual_seed(self.torch_seed)
        if self.arch == 'lstm':
            self.model = LSTMNet(self.D_in, self.H, self.p)

    def fit(self, X, y, verbose=False, checkpoint_fname=None, device='cpu'):

        """
        Train model
        
        Parameters:
        ==========================================================
            X: pd.DataFrame
                input data, should contain tracks and additional covariates
                
            y: np.array
                input response
        """

        torch.manual_seed(self.torch_seed)
        if 'lstm' in self.arch:
            self.model = LSTMNet(self.D_in, self.H, self.p)

        # convert input dataframe to tensors
        X_track = X[self.track_name]  # track
        X_track = torch.tensor(np.array(list(X_track.values)), dtype=torch.float)

        if len(X.columns) > 1:  # covariates
            X_covariates = X[[c for c in X.columns if c != self.track_name]]
            X_covariates = torch.tensor(np.array(X_covariates).astype(float), dtype=torch.float)
        else:
            X_covariates = None

        # response
        y = torch.tensor(y.reshape(-1, 1), dtype=torch.float)

        # initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # initialize dataloader
        if X_covariates is not None:
            dataset = torch.utils.data.TensorDataset(X_track, X_covariates, y)
        else:
            dataset = torch.utils.data.TensorDataset(X_track, y)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        # train_loader = [(X1, X2, y)]

        # train fcnn
        print('fitting dnn...')
        self.model = self.model.to(device)
        for epoch in tqdm(range(self.epochs)):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                # print('shapes input', data[0].shape, data[1].shape)
                if X_covariates is not None:
                    preds = self.model(data[0].to(device), data[1].to(device))
                    y = data[2].to(device)
                else:
                    preds = self.model(data[0].to(device))
                    y = data[1].to(device)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(preds, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if verbose:
                print(f'Epoch: {epoch}, Average loss: {train_loss / len(X_track):.4e}')
            elif epoch % (self.epochs // 10) == 99:
                print(f'Epoch: {epoch}, Average loss: {train_loss / len(X_track):.4e}')
            if checkpoint_fname is not None:
                pkl.dump({'model_state_dict': self.model.state_dict()},
                         open(checkpoint_fname, 'wb'))

    def predict(self, X_new):

        """
        make predictions with new data
        
        Parameters:
        ==========================================================
            X_new: pd.DataFrame
                input new data, should contain tracks and additional covariates
        """
        self.model.eval()
        with torch.no_grad():

            # convert input dataframe to tensors
            X_new_track = X_new[self.track_name]
            X_new_track = torch.tensor(np.array(list(X_new_track.values)), dtype=torch.float)

            if len(X_new.columns) > 1:
                X_new_covariates = X_new[[c for c in X_new.columns if c != self.track_name]]
                X_new_covariates = torch.tensor(np.array(X_new_covariates).astype(float), dtype=torch.float)
                preds = self.model(X_new_track, X_new_covariates)
            else:
                preds = self.model(X_new_track)
        return preds.data.numpy().reshape(1, -1)[0]
