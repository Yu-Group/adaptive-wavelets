import numpy as np
import torch


def run_warmstart(data_loader, trainer, reg_path, epochs=100, 
               out_dir=None):
    '''function to enable warmstart
    Params
    ------
    data_loader: torch.utils.data.DataLoader

    trainer: class
        class to handle training of model
        
    reg_path: list
        list of hyperparameters
        
    epochs: int, optional
        Number of epochs to train the model for each hyperparameter.
    '''
    print('\tWarm starting...')
    for reg_param in reg_path:
        print('\n***Train model at regularization parameter = {}***\n'.format(reg_param))
        trainer.loss_f.lamL1attr = reg_param
        trainer(data_loader, epochs=epochs)
        if out_dir is not None:
            torch.save(trainer.w_transform.state_dict(), out_dir + '{}')
