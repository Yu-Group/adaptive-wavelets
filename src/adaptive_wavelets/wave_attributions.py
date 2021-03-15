import numpy as np
import torch
from torch import nn
from copy import deepcopy

    
class Attributer(nn.Module):
    '''Get attribution scores for wavelet coefficients
    Params
    ------
    mt: nn.Module
        model after all the transformations
    attr_methods: str
        currently support InputXGradient only
    device: str
        use GPU or CPU
    '''    
    def __init__(self, mt, attr_methods='InputXGradient', is_train=True, device='cuda'): 
        super().__init__()
        self.mt = mt.to(device)
        self.attr_methods = attr_methods   
        self.is_train = is_train
        self.device = device
        
    def forward(self, x: tuple, target=1, additional_forward_args=None):
        if self.attr_methods == 'InputXGradient':
            attributions = self.InputXGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'IntegratedGradient':
            attributions = self.IntegratedGradient(x, target, additional_forward_args)
        elif self.attr_methods == 'Saliency':
            attributions = self.Saliency(x, target, additional_forward_args)
        else: 
            raise ValueError
        return attributions
        
    def InputXGradient(self, x: tuple, target=1, additional_forward_args=None):
        outputs = self.mt(x, additional_forward_args)[:,target]
        if self.is_train:
            grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)        
        else:
            grads = torch.autograd.grad(torch.unbind(outputs), x)        
        # input * gradient
        attributions = tuple(xi * gi for xi, gi in zip(x, grads))
        return attributions    
    
    def Saliency(self, x: tuple, target=1, additional_forward_args=None):
        outputs = self.mt(x, additional_forward_args)[:,target]
        if self.is_train:
            grads = torch.autograd.grad(torch.unbind(outputs), x, create_graph=True)        
        else:
            grads = torch.autograd.grad(torch.unbind(outputs), x) 
        return grads
    
    ### TO DO!! ###
    # implement batch version of IG
    def IntegratedGradient(self, x: tuple, target=1, additional_forward_args=None, M=100):
        n = len(x)
        mult_grid = np.array(range(M))/(M-1) # fractions to multiply by

        # compute all the input vecs
        input_vecs = []
        baselines = []
        for i in range(n):
            baselines.append(torch.zeros_like(x[i])) # baseline of zeros
            shape = list(x[i].shape[1:])
            shape.insert(0, M)
            inp = torch.empty(shape, dtype=torch.float32, requires_grad=True).to(self.device)    
            for j, prop in enumerate(mult_grid):
                inp[j] = baselines[i] + prop * (x[i] - baselines[i])
            inp.retain_grad()
            input_vecs.append(inp)

        # run forward pass
        output = self.mt(input_vecs, additional_forward_args)[:,1].sum()
        output.backward(retain_graph=True)

        # ig
        scores = []
        for i in range(n):
            imps = input_vecs[i].grad.mean(0) * (x[i] - baselines[i]) # record all the grads
            scores.append(imps)   
        return tuple(scores)    
    
    
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
    
    
    

