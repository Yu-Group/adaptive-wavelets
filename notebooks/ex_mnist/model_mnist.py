import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

torch.manual_seed(10)

class LinAutoencoder(nn.Module):
    def __init__(self, mid_dim=20, latent_dim=10):
        super(LinAutoencoder, self).__init__()
        ## encoder layers ## 
        self.fc1 = nn.Linear(32*32, mid_dim)
        self.fc2 = nn.Linear(mid_dim, latent_dim)
        
        ## decoder layers ##
        self.fc3 = nn.Linear(latent_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, 32*32)
        
    def transformer(self, x):
        x = x.view(-1, 32*32)
        x = self.fc1(x)
        return x
        
    def encoder(self, x):
        # add layer, with relu activation function
        # and maxpooling after
        x = self.transformer(x)
        x = self.fc2(F.relu(x))
        return x
        
    def decoder(self, x):
        # upsample, followed by a conv layer, with relu activation function  
        # upsample again, output should have a sigmoid applied
        x = F.relu(self.fc3(F.relu(x)))
        x = F.sigmoid(self.fc4(x))    
        x = x.view(-1, 1, 32, 32)
        return x
    
    def i_transformer(self, x):
        x = self.fc2(F.relu(x))
        x = self.decoder(x)
        return x    
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  
        return x    
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.conv3 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)
        
    def transformer(self, x):
        x = self.conv1(x)
        return x
        
    def encoder(self, x):
        # add layer, with relu activation function
        # and maxpooling after
        x = F.relu(self.transformer(x))
        x = F.relu(self.conv2(x)) 
        return x
        
    def decoder(self, x):
        # upsample, followed by a conv layer, with relu activation function  
        # upsample again, output should have a sigmoid applied
        # x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))     
        x = F.sigmoid(self.conv4(x))                  
        return x
    
    def i_transformer(self, x):
        x = F.relu(x)
        x = F.relu(self.conv2(x)) 
        x = self.decoder(x)
        return x        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  
        return x     
    
    
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