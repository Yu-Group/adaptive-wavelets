import torch.nn as nn
import torch.nn.functional as F


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