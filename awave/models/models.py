import torch
import torch.nn.functional as F
from torch import nn


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # hidden layers
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        x1 = x1.unsqueeze(2)  # add input_size dimension (this is usually for the size of embedding vector)
        outputs, (h1, c1) = self.lstm(x1)  # get hidden vec
        h1 = h1.squeeze(0)  # remove dimension corresponding to multiple layers / directions
        if x2 is not None:
            h1 = torch.cat((h1, x2), 1)
        return self.fc(h1)
