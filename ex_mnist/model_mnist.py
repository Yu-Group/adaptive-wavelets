import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


# define net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
#         self.log_softmax = nn.LogSoftmax(dim=1) # might not need this

    def logits(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2_drop(self.conv2(x))))
        x = x.reshape(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
    def forward(self, x):
        # print('forward', x.shape)
        x = self.logits(x)
        # print('later', x.shape)
#         return self.log_softmax(x)
        return F.log_softmax(x, dim=1)

    def predicted_class(self, x):
        pred = self.forward(x)
        _, pred = pred[0].max(0)
        return pred.item() #data[0]


class Net2c(Net):
    def __init__(self):
        super(Net2c, self).__init__()
        self.fc2 = nn.Linear(50, 2)