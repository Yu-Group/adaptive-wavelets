import os,sys
opj = os.path.join
import torch
import torchvision

from .models import CNN, FFN
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_dataloader(root_dir, shuffle=True, pin_memory=True, batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.   

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = torchvision.datasets.MNIST(root=root_dir, 
                                               train=True, 
                                               download=False, 
                                               transform=transformer)
    test_dataset = torchvision.datasets.MNIST(root=root_dir, 
                                              train=False, 
                                              download=False, 
                                              transform=transformer)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, indices=range(20000)),
                                               batch_size=batch_size, 
                                               shuffle=shuffle,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_dataset, indices=range(3000)),
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              pin_memory=pin_memory)
    
    return train_loader, test_loader


def load_pretrained_model(root_dir, device=device):
    """load pretrained model for interpretation
    """     
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(opj(root_dir, 'CNN.pth')))
    cnn = cnn.eval()
    # freeze layers
    for param in cnn.parameters():
        param.requires_grad = False      
    
    ffn = FFN().to(device)
    ffn.load_state_dict(torch.load(opj(root_dir, 'FFN.pth')))
    ffn = ffn.eval()
    # freeze layers
    for param in ffn.parameters():
        param.requires_grad = False      

    return cnn, ffn