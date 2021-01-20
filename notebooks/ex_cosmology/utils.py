import numpy as np
import torch
from torch import nn
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
    