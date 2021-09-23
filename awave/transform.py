import numpy as np
import torch
import torch.nn as nn

from awave.losses import get_loss_f
from awave.utils.train import Trainer


class AbstractWT(nn.Module):

    def fit(self,
            X=None,
            train_loader=None,
            pretrained_model=None,
            lr: float = 0.001,
            num_epochs: int = 20,
            seed: int = 42,
            attr_methods='Saliency',
            target=6,
            lamlSum: float = 1.,
            lamhSum: float = 1.,
            lamL2norm: float = 1.,
            lamCMF: float = 1.,
            lamConv: float = 1.,
            lamL1wave: float = 1.,
            lamL1attr: float = 1.):
        """
        Params
        ------
        X: numpy array or torch.Tensor
            For 1-d signals this should be 3-dimensional, (num_examples, num_curves_per_example, length_of_curve)
            e.g. for 500 1-dimensional curves of length 40 would be (500, 1, 40)
        train_loader: data_loader
            each element should return tuple of (x, _)
        pretrained_model: nn.Module, optional
            pretrained model to distill
        lamlSum : float
            Hyperparameter for penalizing sum of lowpass filter
        lamhSum : float
            Hyperparameter for penalizing sum of highpass filter            
        lamL2norm : float
            Hyperparameter to enforce unit norm of lowpass filter
        lamCMF : float 
            Hyperparameter to enforce conjugate mirror filter   
        lamConv : float
            Hyperparameter to enforce convolution constraint
        lamL1wave : float
            Hyperparameter for penalizing L1 norm of wavelet coeffs
        lamL1attr : float
            Hyperparameter for penalizing L1 norm of attributions
        """
        torch.manual_seed(seed)
        if X is None and train_loader is None:
            raise ValueError('Either X or train_loader must be passed!')
        elif train_loader is None:
            if 'ndarray' in str(type(X)):
                X = torch.Tensor(X).to(self.device)

            # convert to float
            X = X.float()
            if self.wt_type == 'DWT2d':
                X = X.unsqueeze(1)

            # need to pad as if it had y (to match default pytorch dataloaders)
            X = [(X[i], np.nan) for i in range(X.shape[0])]
            train_loader = torch.utils.data.DataLoader(X,
                                                       shuffle=True,
                                                       batch_size=len(X))
        #             print(iter(train_loader).next())
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_f = get_loss_f(lamlSum=lamlSum, lamhSum=lamhSum,
                            lamL2norm=lamL2norm, lamCMF=lamCMF, lamConv=lamConv,
                            lamL1wave=lamL1wave, lamL1attr=lamL1attr)
        trainer = Trainer(pretrained_model,
                          self,
                          optimizer,
                          loss_f,
                          use_residuals=True,
                          target=target,
                          attr_methods=attr_methods,
                          n_print=1, device=self.device)

        # actually train
        self.train()
        trainer(train_loader, epochs=num_epochs)
        self.train_losses = trainer.train_losses
        self.eval()
