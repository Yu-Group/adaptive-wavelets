import torch
from torch import nn
from torchvision import models
from os.path import join as oj

def load_model(model_name='resnet18', device='cuda', inplace=True, data_path='/scratch/users/vision/data/cosmo'):
    '''Load a pretrained model and make shape alterations for cosmology
    '''
    
    # Modifying the model to predict the three cosmological parameters from single channel images
    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 3)
        
        if inplace == False:
            mods = list(model_ft.modules())
            for mod in mods: 
                t = str(type(mod))
                if 'ReLU' in t:
                    mod.inplace = False   

            model_ft = model_ft.to(device)
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