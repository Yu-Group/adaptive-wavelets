import numpy as np
import torch
import random
from copy import deepcopy


def get_tensors(data_loader):
    """Given dataloader return inputs and labels in torch.Tensor
    """
    inputs, labels = data_loader.dataset.tensors
    X = deepcopy(inputs)
    y = deepcopy(labels)    
    return (X, y)


def max_fun(X, sgn="abs", m=1):
    """Given an array X return maximum values across columns for every row
    """
    if sgn == "abs":
        Y = abs(X)
    elif sgn == "neg":
        Y = -X
    elif sgn == "pos":
        Y = X
    else:
        print('no such sign supported')
    id_s = np.argsort(Y, axis=1)[:,::-1]
    index = id_s[:,:m]
    return np.take_along_axis(X, index, axis=1)


def max_transformer(w_transform, 
                    train_loader, 
                    test_loader,
                    sgn="abs", 
                    m=1):
    """Compute maximum features of wavelet representations across all scales 
    """
    w_transform = w_transform.to('cpu')
    J = w_transform.J
    
    # transform train data
    (Xs, y) = get_tensors(train_loader)
    X = []
    data_t = w_transform(Xs)
    for j in range(J+1):
        d = data_t[j].detach().squeeze().numpy()
        X.append(max_fun(d, sgn=sgn, m=m))
    X = np.hstack(X)
    y = y.detach().squeeze().numpy()
    
    # transform test data
    (Xs_test, y_test) = get_tensors(test_loader)
    X_test = []
    data_t = w_transform(Xs_test)
    for j in range(J+1):
        d = data_t[j].detach().squeeze().numpy()
        X_test.append(max_fun(d, sgn=sgn, m=m))
    X_test = np.hstack(X_test)
    y_test = y_test.detach().squeeze().numpy()    
    
    return (X, y), (X_test, y_test)  
