import numpy as np
import imageio
import logging
import os, sys
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F

from disvae.utils.modelIO import save_model

# trim modules
sys.path.append('../../trim')
from trim import TrimModel, DecoderEncoder
from captum.attr import *
from copy import deepcopy

# TO-DO: currently support two attr methods
ATTR_METHOS = ['InputXGradient'] 


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True,
                 classifier=None,
                 trim_lamb=0.0,
                 attr_lamb=0.0):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info("Training Device: {}".format(self.device))
        self.classifier = classifier
        self.trim_lamb = trim_lamb
        self.attr_lamb = attr_lamb
        self.L1Loss = torch.nn.L1Loss()
        self.L2Loss = torch.nn.MSELoss()
        if self.classifier is not None and self.trim_lamb > 0:
            self._prepend_transformation()
        if self.attr_lamb > 0:
            self._create_latent_map()

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))
        
    def _prepend_transformation(self):
        """
        Prepends transformation onto network.
        
        Parameters
        ----------
        """
        transform_i = lambda s: self.model.decoder(s)
        m_t = TrimModel(self.classifier, transform_i, use_residuals=True)
        self.attributer = InputXGradient(m_t)
        
    def _create_latent_map(self):
        """
        Create saliency object for decoder-encoder map.
        
        Parameters
        ----------
        """
        self.latent_map = DecoderEncoder(self.model)

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, data in enumerate(data_loader):
                iter_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        inputs, labels = data
        batch_size, channel, height, width = inputs.size()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(inputs)
            loss = self.loss_f(inputs, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)
            # penalize trim score
            if self.classifier is not None and self.trim_lamb > 0:
                s = deepcopy(latent_dist[0].detach())
                attributions = self.attributer.attribute(s, target=labels, additional_forward_args=deepcopy(inputs))
                loss += self.trim_lamb * self.L1Loss(attributions, torch.zeros_like(attributions))    
            # penalize change in one attribute wrt the other attributes
            if self.attr_lamb > 0:
                s = deepcopy(latent_dist[0].detach())
                s = s.requires_grad_(True)
                s_output = self.latent_map(s)
                for i in range(self.model.latent_dim):
                    col_idx = np.arange(self.model.latent_dim)!=i
                    gradients = torch.autograd.grad(s_output[:,i], s, grad_outputs=torch.ones_like(s_output[:,i]), 
                                                    retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients_pairwise = gradients[:,col_idx]
                    loss += self.attr_lamb * self.L2Loss(gradients_pairwise, torch.zeros_like(gradients_pairwise))                                                    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(inputs, self.model, self.optimizer, storer)

        return loss.item()  
    

class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
