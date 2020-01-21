import torch
from torch import nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, p):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(p, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)
        self.use_softmax = True

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x