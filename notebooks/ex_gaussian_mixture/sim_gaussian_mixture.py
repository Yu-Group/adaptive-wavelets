import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
# import acd
from random import randint
from copy import deepcopy
import pickle as pkl
import argparse

sys.path.append('../../src/vae')
sys.path.append('../../src/dsets/gaussian_mixture')
sys.path.append('../../lib/trim')
from model import init_specific_model
from losses import Loss
from dset import get_dataloaders
from training import Trainer
from utils import traversals

# trim modules
from trim import DecoderEncoder

parser = argparse.ArgumentParser(description='Gaussian Mixture Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hidden_dim', type=int, default=12,
                   help='number of hidden variables in VAE (default: 12)')
parser.add_argument('--beta', type=float, default=1,
                   help='weight of the KL term')
parser.add_argument('--mu', type=float, default=0,
                   help='weight of the mu term')
parser.add_argument('--lamPT', type=float, default=0,
                   help='weight of the pointwise local indepndence term')
parser.add_argument('--lamCI', type=float, default=0,
                   help='weight of the conditional local indepndence term')
parser.add_argument('--lamNN', type=float, default=0,
                   help='weight of the nearest-neighbor term')
parser.add_argument('--lamH', type=float, default=0,
                   help='weight of the Hessian term')
parser.add_argument('--alpha', type=float, default=0,
                   help='weight of the mutual information term')
parser.add_argument('--gamma', type=float, default=0,
                   help='weight of the dim-wise KL term')
parser.add_argument('--tc', type=float, default=0,
                   help='weight of the total correlation term')
parser.add_argument('--dirname', default='vary',
                   help='name of directory')
parser.add_argument('--warm_start', default=None,
                   help='name of hyperparameter to run warm start model (default: None)')
parser.add_argument('--seq_init', type=float, default=0,
                   help='An initial value for a sequence of varying parameters')


class p:
    '''Parameters for Gaussian mixture simulation
    '''
    # parameters for generating data
    train_n_samples_per_cluster = 5000
    test_n_samples_per_cluster = 1000
    latent_means = [[0.0, 0.0], [15.0, 15.0], [30.0, 30.0]]
    latent_vars = [[4, 1], [4, 1], [4, 1]]
    noise_dim = 8
    noise_var = 0.01
    
    # parameters for model architecture
    orig_dim = 10
    latent_dim = 4    
    hidden_dim = 12
    
    # parameters for training
    train_batch_size = 64
    test_batch_size = 100
    lr = 5e-4
    beta = 0.0
    mu = 0.0
    lamPT = 0.0
    lamCI = 0.0
    lamNN = 0.0
    lamH = 0.0
    alpha = 0.0
    gamma = 0.0
    tc = 0.0
    num_epochs = 100
    
    seed = 13
    warm_start = None # which parameter to warm start with respect to
    seq_init = 1      # value of warm_start parameter to start with respect to
    
    # SAVE MODEL
#     out_dir = "/home/ubuntu/local-vae/notebooks/ex_gaussian_mixture/results" # wooseok's setup
    out_dir = '/scratch/users/vision/chandan/local-vae' # chandan's setup
    dirname = "vary"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

    def _str(self):
        vals = vars(p)
        return 'beta=' + str(vals['beta']) + '_mu=' + str(vals['mu']) + '_lamPT=' + str(vals['lamPT']) + '_lamCI=' + str(vals['lamCI']) + '_seed=' + str(vals['seed']) \
                + '_hdim=' + str(vals['hidden_dim']) + '_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
class s:
    '''Parameters to save
    '''
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
# generate data
def define_dataloaders(p):
    """A generic data loader
    """
    train_loader, train_latents = get_dataloaders(n_samples_per_cluster=p.train_n_samples_per_cluster, 
                                                  latent_means=p.latent_means,
                                                  latent_vars=p.latent_vars,
                                                  extra_dim=p.noise_dim, 
                                                  var=p.noise_var,
                                                  batch_size=p.train_batch_size,
                                                  shuffle=True,
                                                  return_latents=True) 
    test_loader, test_latents = get_dataloaders(n_samples_per_cluster=p.test_n_samples_per_cluster, 
                                                latent_means=p.latent_means,
                                                latent_vars=p.latent_vars,
                                                extra_dim=p.noise_dim, 
                                                var=p.noise_var,
                                                batch_size=p.test_batch_size, 
                                                shuffle=False,
                                                return_latents=True)   
    return((train_loader, train_latents), (test_loader, test_latents))


# calculate losses
def calc_losses(model, data_loader, loss_f):
    """
    Tests the model for one epoch.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader

    loss_f: loss object

    Return
    ------
    """    
    model.eval()
    n_data = data_loader.dataset.data.shape[0]
    rec_loss = 0
    kl_loss = 0
    mu_loss = 0
    mi_loss = 0
    tc_loss = 0
    dw_kl_loss = 0
    pt_loss = 0
    ci_loss = 0
    nearest_neighbor_loss = 0
    hessian_loss = 0

    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        recon_data, latent_dist, latent_sample = model(data)
        latent_map = DecoderEncoder(model, use_residuals=True)
        latent_output = latent_map(latent_sample, data)
        _ = loss_f(data, recon_data, latent_dist, latent_sample, n_data, latent_output, model.decoder) 
        rec_loss += loss_f.rec_loss.item()
        kl_loss += loss_f.kl_loss.item()
        mu_loss += loss_f.mu_loss.item()
        mi_loss += loss_f.mi_loss.item()
        tc_loss += loss_f.tc_loss.item()
        dw_kl_loss += loss_f.dw_kl_loss.item()
        pt_loss += loss_f.pt_loss.item() if type(loss_f.pt_loss) == torch.Tensor else 0
        ci_loss += loss_f.ci_loss.item()if type(loss_f.ci_loss) == torch.Tensor else 0
        nearest_neighbor_loss += loss_f.nearest_neighbor_loss.item()if type(loss_f.nearest_neighbor_loss) == torch.Tensor else 0        
        hessian_loss += loss_f.hessian_loss.item()if type(loss_f.hessian_loss) == torch.Tensor else 0        
        

    n_batch = batch_idx + 1
    rec_loss /= n_batch
    kl_loss /= n_batch
    mu_loss /= n_batch
    mi_loss /= n_batch
    tc_loss /= n_batch
    dw_kl_loss /= n_batch
    pt_loss /= n_batch
    ci_loss /= n_batch
    nearest_neighbor_loss /= n_batch
    hessian_loss /= n_batch

    return (rec_loss, kl_loss, mu_loss, mi_loss, tc_loss, dw_kl_loss, pt_loss, ci_loss, nearest_neighbor_loss, hessian_loss)
    
    
def measure_angle_iteration(model, data):
    batch_size, dim = data.shape
    
    results = []
    for batch_idx in range(batch_size):
        data_i = data[batch_idx:batch_idx+1]
        decoded_traversal = traversals(model, data=data_i, n_latents=p.latent_dim)[:,:2]
        
        # find 2 latents corresponding to the highest variance in the original space
        variab = []
        for i in range(p.latent_dim):
            # get x corresponding to 100 small changes in z
            x = decoded_traversal[100*i:100*(i+1)]
            
            # calculate variance in x space
            tot_var = torch.var(x[:,0]) + torch.var(x[:,1])
            variab.append(tot_var.item())
        variab = torch.Tensor(variab)
        _, idxs = torch.sort(variab, descending=True)
        idxs = idxs[:2]

        # find the minimum angle of each latent direction with the x and y axis
        angles = []
        for i in range(2):
            x = decoded_traversal[100*idxs[i]:100*(idxs[i]+1)]
            v = x[-1] - x[0]
            if torch.norm(v) > 0:
                angles.append(abs(v/torch.norm(v)))
        if len(angles) == 2:
            # angles[0, 0] is  1st latent var, x-axis
            # angles[0, 1] is  1st latent var, y-axis
            angles = torch.stack(angles)
            
            # is 1st aligned with x and 2nd aligned with y?
            s1 = torch.sqrt((angles[0,0] - 1)**2 + (angles[1,1] - 1)**2)
            
            # is 2nd aligned with x and 1st aligned with y?
            s2 = torch.sqrt((angles[0,1] - 1)**2 + (angles[1,0] - 1)**2)
            
            # return minimum of those two
            results.append(torch.min(s1, s2))

    return torch.stack(results)


def calc_disentangle_metric(model, data_loader):
    '''Returns disentanglement metric
    Smaller is better (closer to capturing groundtruth axes)
    '''
    model.eval()
    
    dis_metric = []
    for _, data in enumerate(data_loader):
        results = measure_angle_iteration(model, data)
        dis_metric.append(results)
        
    return torch.cat(dis_metric)


def warm_start(p, out_dir):
    '''load results and initialize model where beta=p.beta, mu=p.mu
    '''
    print('\twarm starting...')
    fnames = sorted(os.listdir(out_dir))
    params = []
    models = []
    for fname in fnames:
        if f'beta={p.beta}' in fname and f'mu={p.mu}' in fname:
            if fname[-3:] == 'pkl':
                result = pkl.load(open(opj(out_dir, fname), 'rb'))
                params.append(result[p.warm_start])
            if fname[-3:] == 'pth':
                m = init_specific_model(orig_dim=p.orig_dim, 
                                        latent_dim=p.latent_dim, 
                                        hidden_dim=p.hidden_dim).to(device)
                m.load_state_dict(torch.load(opj(out_dir, fname)))
                models.append(m)
    max_idx = np.argmax(np.array(params))
    model = models[max_idx]
    return model
    
if __name__ == '__main__':
    args = parser.parse_args()
    for arg in vars(args):
        setattr(p, arg, getattr(args, arg))
    
    # create dir
    out_dir = opj(p.out_dir, p.dirname)
    os.makedirs(out_dir, exist_ok=True)        

    # seed
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    # get dataloaders
    (train_loader, train_latents), (test_loader, test_latents) = define_dataloaders(p)

    # prepare model
    # optimize model with warm start
    # should have already trained model in this directory with p.warm_start parameter set to p.seq_init
    if p.warm_start is not None and eval('p.' + p.warm_start) > p.seq_init:
        model = warm_start(p, out_dir)        
    else:
        model = init_specific_model(orig_dim=p.orig_dim, latent_dim=p.latent_dim, hidden_dim=p.hidden_dim)
        model = model.to(device)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    loss_f = Loss(beta=p.beta, mu=p.mu, lamPT=p.lamPT, lamCI=p.lamCI,
                  lamNN=p.lamNN,
                  alpha=p.alpha, gamma=p.gamma, tc=p.tc, is_mss=True)
    trainer = Trainer(model, optimizer, loss_f, device=device)
    trainer(train_loader, test_loader, epochs=p.num_epochs)
    
    # calculate losses
    print('calculating losses and metric...')    
    rec_loss, kl_loss, mu_loss, mi_loss, tc_loss, dw_kl_loss, \
    pt_loss, ci_loss, nearest_neighbor_loss, hessian_loss = calc_losses(model, test_loader, loss_f)
    s.reconstruction_loss = rec_loss
    s.kl_normal_loss = kl_loss
    s.mu_squared_loss = mu_loss
    s.total_correlation = tc_loss
    s.mutual_information = mi_loss
    s.dimensionwise_kl_loss = dw_kl_loss
    s.pt_local_independence_loss = pt_loss
    s.ci_local_independence_loss = ci_loss
    s.disentanglement_metric = calc_disentangle_metric(model, test_loader).mean().item()
    s.nearest_neighbor_loss = nearest_neighbor_loss
    s.hessian_loss = hessian_loss
    s.net = model    
    
    # save
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, p._str(p) + '.pkl'), 'wb'))    
    torch.save(model.state_dict(), opj(out_dir, p._str(p) + '.pth')) 