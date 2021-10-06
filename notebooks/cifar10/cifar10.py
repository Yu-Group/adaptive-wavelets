import torch
import torchvision
import os
opj = os.path.join



def get_dataset(root_dir='../../data', batch_size=64, **kwargs):
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, 
                                                 train=True, 
                                                 download=True, 
                                                 transform=transformer)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, 
                                                train=False, 
                                                download=True, 
                                                transform=transformer)
    return train_dataset, test_dataset
    

def get_dataloader(root_dir='../../data', shuffle=True, pin_memory=True, batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.   

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    train_dataset, test_dataset = get_dataset(root_dir='../../data', batch_size=batch_size, **kwargs)
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=shuffle,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              pin_memory=pin_memory)
    
    return train_loader, test_loader

def get_batch(batch_size=1, train=False):
    train_loader, test_loader = get_dataloader(batch_size=batch_size)
    if train:
        return next(iter(train_loader))
    else:
        return next(iter(test_loader))
    
def create_dataloader(X, y, shuffle=True, pin_memory=True, batch_size=64):
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                               batch_size=batch_size, 
                                               shuffle=shuffle,
                                               pin_memory=pin_memory)
    return loader