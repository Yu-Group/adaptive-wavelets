import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from os.path import join as oj

def load_model(model_name='resnet18', device='cuda', num_params=3, inplace=True, data_path='/scratch/users/vision/data/cosmo'):
    '''Load a pretrained model and make shape alterations for cosmology
    '''
    
    # Modifying the model to predict the three cosmological parameters from single channel images
    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_params)
        
        if inplace == False:
            mods = list(model_ft.modules())
            for mod in mods: 
                t = str(type(mod))
                if 'ReLU' in t:
                    mod.inplace = False   

        model_ft = model_ft.to(device)
        if data_path is not None:
            model_ft.load_state_dict(torch.load(oj(data_path, 'resnet18_state_dict')))

    elif model_name == 'vgg16':
        model_ft = models.vgg16(pretrained=False)
        model_ft.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = 4096 # model_ft.fc.n_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load(oj(data_path, 'vgg16_adam_9_0.012')))
        
    model_ft.eval()
    return model_ft


class AutoEncoder(nn.Module):
    def __init__(self, img_size=(1,64,64)):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        encoder : torch.nn.Module
            class of encoder
        
        decoder : torch.nn.Module
            class of decoder
        """
        super(AutoEncoder, self).__init__()
        
        # Layer parameters
        hid_channels = 3
        kernel_size = 4
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        
        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), latent_dim)
        
        # Fully connected layers
        self.linT1 = nn.Linear(latent_dim, np.product(self.reshape))         
        
        # Transpose Convolutional layers
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)
      
        
    def encoder(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        return x
    
    def decoder(self, x):
        batch_size = x.size(0)
        
        x = torch.relu(self.convT1(x))
        x = self.convT2(x)

        return x       
    

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_sample = self.encoder(x)
        reconstruct = self.decoder(latent_sample)
        
        return reconstruct, latent_sample
    
    
class AutoEncoderSimple(nn.Module):
    def __init__(self, img_size=(1,256,256), hid_channels=2):
        """
        Class which defines model and forward pass. Consists of one layer of Conv and ConvTranspose.
        Parameters
        ----------
        img_size : tuple of (C,H,W)
        
        hid_channels : int
            number of hidden channels
        """
        super(AutoEncoderSimple, self).__init__()
        
        # Layer parameters
        kernel_size = 4
        n_chan = img_size[0]
        self.img_size = img_size

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)       

        # Transpose Convolutional layers
        self.convT1 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)
        
    def encoder(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = self.conv1(x)
        x = torch.relu(x)
        
        return x
    
    def decoder(self, x):
        batch_size = x.size(0)
        
        x = self.convT1(x)
#         x = torch.tanh(x)

        return x            

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_sample = self.encoder(x)
        reconstruct = self.decoder(latent_sample)
        
        return reconstruct, latent_sample  
    
    
class AutoEncoderMix(nn.Module):
    def __init__(self, img_size=(1,64,64), hid_channels_s=5, hid_channels_d=5, latent_dim=20):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        encoder : torch.nn.Module
            class of encoder
        
        decoder : torch.nn.Module
            class of decoder
        """
        super(AutoEncoderMix, self).__init__()
        
        # Layer parameters
        kernel_size = 4
        self.img_size = img_size
        n_chan = self.img_size[0]
        self.reshape = (hid_channels_d, 16, 16)

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1s = nn.Conv2d(n_chan, hid_channels_s, kernel_size, **cnn_kwargs)
        self.conv1d = nn.Conv2d(n_chan, hid_channels_d, kernel_size, **cnn_kwargs)
        self.conv2d = nn.Conv2d(hid_channels_d, hid_channels_d, kernel_size, **cnn_kwargs)
        
        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), latent_dim)
        
        # Fully connected layers
        self.linT1 = nn.Linear(latent_dim, np.product(self.reshape))         
        
        # Transpose Convolutional layers
        self.convT1s = nn.ConvTranspose2d(hid_channels_s, n_chan, kernel_size, **cnn_kwargs)
        self.convT1d = nn.ConvTranspose2d(hid_channels_d, hid_channels_d, kernel_size, **cnn_kwargs)
        self.convT2d = nn.ConvTranspose2d(hid_channels_d, n_chan, kernel_size, **cnn_kwargs)
        
        
    def encoder_s(self, x):
        x = self.conv1s(x)
        return x
    

    def decoder_s(self, x):
        x = self.convT1s(x)
        return x
    
    
    def encoder_d(self, x):
        batch_size = x.size(0)
        
        x = torch.relu(self.conv1d(x))
        x = torch.relu(self.conv2d(x))
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        return x
    
    
    def decoder_d(self, x):
        batch_size = x.size(0)
        
        x = torch.relu(self.linT1(x))
        x = x.view(batch_size, *self.reshape)
        x = torch.relu(self.convT1d(x))
        x = self.convT2d(x)
        return x   
    

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_sample1, latent_sample2 = self.encoder_s(x), self.encoder_d(x)
        reconstruct1 = self.decoder_s(latent_sample1) 
        reconstruct2 = self.decoder_d(latent_sample2)
        
        return reconstruct1 + reconstruct2, [reconstruct1, reconstruct2]


class AutoEncoderModelBased(nn.Module):
    def __init__(self, img_size=(1,64,64), m=None):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        encoder : torch.nn.Module
            class of encoder
        
        decoder : torch.nn.Module
            class of decoder
        """
        super(AutoEncoderModelBased, self).__init__()
        
        # Layer parameters
        kernel_size = 4
        self.img_size = img_size
        n_chan = self.img_size[0]
        self.m = m

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(512, 256, kernel_size, **cnn_kwargs)  
        self.convT2 = nn.ConvTranspose2d(256, 128, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(128, 64, kernel_size, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(64, 32, kernel_size, **cnn_kwargs)
        self.convT5 = nn.ConvTranspose2d(32, 16, kernel_size, **cnn_kwargs)
        self.convT6 = nn.ConvTranspose2d(16, 1, kernel_size, **cnn_kwargs)

        
    def encoder(self, x):
        batch_size = x.size(0)

        x = self.m.conv1(x)
        x = self.m.bn1(x)
        x = self.m.relu(x)
        x = self.m.maxpool(x)
        x = self.m.layer1(x)
        x = self.m.layer2(x)
        x = self.m.layer3(x)
        x = self.m.layer4(x)
        x = self.m.avgpool(x)
        
        return x
    
    def decoder(self, x):
        batch_size = x.size(0)
        
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT4(x))
        x = torch.relu(self.convT5(x))
        x = self.convT6(x)

        return x            

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_sample = self.encoder(x)
        reconstruct = self.decoder(latent_sample)
        
        return reconstruct, latent_sample    