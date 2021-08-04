import os,sys
opj = os.path.join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
from copy import deepcopy
sys.path.append('../../data/biology')
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import torch.utils.data as data_utils
import pickle as pkl
# from preprocessing import neural_net_sklearn
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
    
    def __init__(self, D_in=40, H=40, p=17, epochs=1000, batch_size=100, track_name='X_same_length_normalized', arch='fcnn', torch_seed=2):
        
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
        if self.arch == 'fcnn':
            self.model = models.FCNN(self.D_in, self.H, self.p)
        elif 'lstm' in self.arch:
            self.model = models.LSTMNet(self.D_in, self.H, self.p)
        elif 'cnn' in self.arch:
            self.model = models.CNN(self.D_in, self.H, self.p)
        elif 'attention' in self.arch:
            self.model = models.AttentionNet(self.D_in, self.H, self.p)   
        elif 'video' in self.arch:
            self.model = models.VideoNet()


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
        if self.arch == 'fcnn':
            self.model = models.FCNN(self.D_in, self.H, self.p)
        elif 'lstm' in self.arch:
            self.model = models.LSTMNet(self.D_in, self.H, self.p)
        elif 'cnn' in self.arch:
            self.model = models.CNN(self.D_in, self.H, self.p)
        elif 'attention' in self.arch:
            self.model = models.AttentionNet(self.D_in, self.H, self.p)   
        elif 'video' in self.arch:
            self.model = models.VideoNet()
        
        # convert input dataframe to tensors
        X_track = X[self.track_name] # track
        X_track = torch.tensor(np.array(list(X_track.values)), dtype=torch.float)
        
        if len(X.columns) > 1: # covariates
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
        #train_loader = [(X1, X2, y)]
        
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
                print(f'Epoch: {epoch}, Average loss: {train_loss/len(X_track):.4e}')
            elif epoch % (self.epochs // 10) == 99:
                print(f'Epoch: {epoch}, Average loss: {train_loss/len(X_track):.4e}')
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
        
        
class VideoNet(nn.Module):
    def __init__(self):

        super(VideoNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
        self.lstm = nn.LSTM(input_size=1, hidden_size=40, num_layers=1, batch_first=True)
        self.fc = nn.Linear(40, 1) 
#         self.conv2 = nn.Conv1d(in_channels=H, out_channels=3, kernel_size=5)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=2)
#         self.fc = nn.Linear(18 + p, 1) # this is hard-coded
    
    def forward(self, x):
        '''
        x: torch.Tensor
            (batch_size, time_steps, height, width)
          = (batch_size, 40, 10, 10)
        '''
#         print('in shape', x.shape)
        # extract features from each time_step separately
        # reshape time_steps and batch into same dim
        batch_size = x.shape[0]
        T = x.shape[1]
        x = x.reshape(batch_size * T, 1, x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.max(x, dim=3).values
        x = torch.max(x, dim=2).values
        
        # extract time_steps back out
        # run lstm on result 1D time series
        x = x.reshape(batch_size, T, 1)
        outputs, (h1, c1) = self.lstm(x) # get hidden vec
        h1 = h1.squeeze(0) # remove dimension corresponding to multiple layers / directions
        return self.fc(h1)

class FCNN(nn.Module):
    
    """
    customized (one hidden layer) fully connected neural network class
    """

    def __init__(self, D_in, H, p):
        
        """
        Parameters:        
        ==========================================================
            D_in: int
                dimension of input track
                
            H: int
                hidden layer size
                
            p: int
                number of additional covariates (such as lifetime, msd, etc..., to be concatenated to the hidden layer)            
        """

        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        #self.fc2 = nn.Linear(H, H)
        self.bn1 = nn.BatchNorm1d(H)
        self.fc2 = nn.Linear(H + p, 1) 
    
    def forward(self, x1, x2):
        
        z1 = self.fc1(x1)
        z1 = self.bn1(z1)
        h1 = F.relu(z1)
        if x2 is not None:
            h1 = torch.cat((h1, x2), 1)
        z2 = self.fc2(h1)
        #h2 = F.relu(z2)
        #z3 = self.fc3(h2)       
        
        return z2
    
    
class LSTMNet(nn.Module):
    def __init__(self, D_in, H, p):
        
        """
        Parameters:        
        ==========================================================
            D_in: int
                dimension of input track (ignored, can be variable)
                
            H: int
                hidden layer size
                
            p: int
                number of additional covariates (such as lifetime, msd, etc..., to be concatenated to the hidden layer)            
        """

        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=H, num_layers=1, batch_first=True)
        self.fc = nn.Linear(H + p, 1) 
    
    def forward(self, x1, x2=None):
        x1 = x1.unsqueeze(2) # add input_size dimension (this is usually for the size of embedding vector)
        outputs, (h1, c1) = self.lstm(x1) # get hidden vec
        h1 = h1.squeeze(0) # remove dimension corresponding to multiple layers / directions
        if x2 is not None:
            h1 = torch.cat((h1, x2), 1)
        return self.fc(h1)
    
class CNN(nn.Module):
    def __init__(self, D_in, H, p):
        
        """
        Parameters:        
        ==========================================================
            D_in: int
                dimension of input track (ignored, can be variable)
                
            H: int
                hidden layer size
                
            p: int
                number of additional covariates (such as lifetime, msd, etc..., to be concatenated to the hidden layer)            
        """

        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=H, kernel_size=7)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=H, out_channels=3, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(18 + p, 1) # this is hard-coded
    
    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1) # add channel dim
        x1 = self.conv1(x1)
        x1 = self.maxpool1(x1)
        x1 = self.conv2(x1)
        x1 = self.maxpool2(x1)
        x1 = x1.reshape(x1.shape[0], -1) # flatten channel dim
        
        if x2 is not None:
            x1 = torch.cat((x1, x2), 1)
        return self.fc(x1)
    
class AttentionNet(nn.Module):
    
    """
    customized (one hidden layer) fully connected neural network class
    """

    def __init__(self, D_in, H, p):
        
        """
        Parameters:        
        ==========================================================
            D_in: int
                dimension of input track (ignored, can be variable)
                
            H: int
                hidden layer size
                
            p: int
                number of additional covariates (such as lifetime, msd, etc..., to be concatenated to the hidden layer)            
        """

        super(AttentionNet, self).__init__()
        self.att1 = nn.MultiheadAttention(embed_dim=18, num_heads=3)
        self.ln1 = nn.LayerNorm(D_in)
        self.fc1 = nn.Linear(D_in, 1) 
        self.relu1 = nn.ReLU()
        self.att2 = nn.MultiheadAttention(embed_dim=18, num_heads=3)
        self.ln2 = nn.LayerNorm(D_in)
        self.fc2 = nn.Linear(D_in + p, 1) 
    
    def forward(self, x1, x2):
        print(x1.shape)
        x1 = self.att1(x1, x1)
        x1 = self.ln1(x1)
        x1 = self.fc1(x1)
        x1 = self.relu1(x1)
        x1 = self.att2(x1, x1)
        x1 = self.ln2(x1)
        
        if x2 is not None:
            h1 = torch.cat((h1, x2), 1)
        return self.fc2(h1)

class MaxLinear(nn.Module):
    '''Takes flattened input and predicts it using many linear units
        X: batch_size x num_timepoints
    '''

    def __init__(self, input_dim=24300, num_units=20, nonlin=F.relu, use_bias=False):
        super(MaxLinear, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_units, bias=use_bias)

    #         self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
        #         print('in shape', X.shape, X.dtype)
        X = self.fc1(X)  # .max(dim=-1)
        #         print('out shape', X.shape, X.dtype)
        X = torch.max(X, dim=1)[0]  # 0 because this returns max, indices
        #         print('out2 shape', X.shape, X.dtype)
        return X  # + self.offset


class MaxConv(nn.Module):
    '''Takes flattened input and predicts it using many conv unit
        X: batch_size x 1 x num_timepoints
            OR
        X: list of size (num_timepoints,)
    '''

    def __init__(self, num_units=20, kernel_size=30, nonlin=F.relu, use_bias=False):
        super(MaxConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_units, kernel_size=kernel_size, bias=use_bias)
        #         torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
        if type(X) == list:
            print('list')
            X = torch.tensor(np.array(X).astype(np.float32))
            X = X.unsqueeze(0)
            X = X.unsqueeze(0)
            print(X.shape)
        #         print('in shape', X.shape, X.dtype)
        else:
            X = X.unsqueeze(1)
        X = self.conv1(X)  # .max(dim=-1)
        #         print('out shape', X.shape, X.dtype)
        # max over channels
        X = torch.max(X, dim=1)[0]  # 0 because this returns max, indices

        # max over time step
        X = torch.max(X, dim=1)[0] + self.offset  # 0 because this returns max, indices
        #         print('out2 shape', X.shape, X.dtype)

        X = X.unsqueeze(1)

        #         print('preds', X)
        return X


class MaxConvLinear(nn.Module):
    '''Takes input patch, uses linear filter to convert it to time series, then runs temporal conv, then takes max
        X: batch_size x H_patch x W_patch x time
    '''

    def __init__(self, num_timepoints=300, num_linear_filts=1, num_conv_filts=3, patch_size=9,
                 kernel_size=30, nonlin=F.relu, use_bias=False):
        super(MaxConvLinear, self).__init__()
        self.fc1 = nn.Linear(patch_size * patch_size, num_linear_filts, bias=use_bias)
        self.conv1 = nn.Conv1d(in_channels=num_linear_filts, out_channels=num_conv_filts, kernel_size=kernel_size,
                               bias=use_bias)
        self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
        s = X.shape  # batch_size x H_patch x W_patch x time
        X = X.reshape(s[0], s[1] * s[2], s[3])
        X = torch.transpose(X, 1, 2)
        #         print('in shape', X.shape, X.dtype)
        X = self.fc1(X)  # .max(dim=-1)
        X = torch.transpose(X, 1, 2)

        X = self.conv1(X)  # .max(dim=-1)
        #         print('out shape', X.shape, X.dtype)
        # max over channels
        X = torch.max(X, dim=1)[0]  # 0 because this returns max, indices

        # max over time step
        X = torch.max(X, dim=1)[0]  # + self.offset # 0 because this returns max, indices
        #         print('out2 shape', X.shape, X.dtype)

        X = X.unsqueeze(1)
        return X
