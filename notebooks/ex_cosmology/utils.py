import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage.transform import rescale
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
def tuple_Sum(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(x[i])
    return output/num    


def tuple_L1Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(abs(x[i]))
    return output/num


def tuple_L2Loss(x):
    output = 0
    num = len(x)
    for i in range(num):
        output += torch.sum(x[i]**2)
    return output/num   


def compute_tuple_dim(x):
    tot_dim = 0
    for i in range(len(x)):
        shape = torch.tensor(x[i].shape)
        tot_dim += torch.prod(shape).item()
    return tot_dim


def pad_within(x, stride=2, start_row=0, start_col=0):
    w = x.new_zeros(stride, stride)
    if start_row == 0 and start_col == 0:
        w[0, 0] = 1
    elif start_row == 0 and start_col == 1:
        w[0, 1] = 1
    elif start_row == 1 and start_col == 0:
        w[1, 0] = 1
    else:
        w[1, 1] = 1
    if len(x.shape) == 2:
        x = x[None,None]
    return F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1)).squeeze()


def thresh_attrs(attributions: tuple, sp_level):
    """ 
    Threshold attributions retaining those with top absolute attributions 
    """    
    batch_size = attributions[0].shape[0]
    J = len(attributions)
    b = torch.tensor([]).to(device)
    list_of_size = [0]    
    for j in range(J):
        a = abs(attributions[j]).reshape(batch_size, -1)
        if j == 0:
            b = deepcopy(a.detach())
        else:
            b = torch.cat((b,a), dim=1)
        list_of_size.append(list_of_size[-1] + a.shape[1])
    sort_indexes = torch.argsort(b, dim=1, descending=True)      
    
    m = torch.zeros_like(sort_indexes)
    for i in range(batch_size):
        m[i][sort_indexes[i,:sp_level]] = 1

    list_of_masks = []
    for j in range(J):
        n0 = list_of_size[j]
        n1 = list_of_size[j+1]
        list_of_masks.append(m[:,n0:n1].reshape(attributions[j].shape))

    output = []
    for j in range(J):
        output.append(torch.mul(list_of_masks[j], attributions[j]))
    return tuple(output)       
    
    
def viz_list(x: list, figsize=(10, 10), scale=2):
    '''Plot images in the list
    Params
    ------
    x: list
        list of images
    figsize: tuple
        figure size    
    '''
    ls = len(x)
    x_min = 1e4
    x_max = -1e4
    for i in range(ls):
        x_min = min(x[i].min(), x_min)
        x_max = max(x[i].max(), x_max)
        
    plt.figure(figsize=figsize, dpi=200)
    for i in range(ls):
        plt.subplot(1, ls, i + 1)
        plt.imshow(rescale(x[i], scale, mode='constant'), cmap='gray', extent=[0,2,0,2], vmin=x_min, vmax=x_max)
        plt.axis('off')  
    plt.tight_layout()
    plt.show()    