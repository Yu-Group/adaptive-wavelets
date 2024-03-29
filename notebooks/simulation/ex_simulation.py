import os
import random

import numpy as np
import torch

opj = os.path.join
import pickle as pkl
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# adaptive-wavelets modules

from awave.losses import get_loss_f
from awave.utils.train import Trainer
from awave.evaluate import Validator
from awave.transform1d import DWT1d
from awave.data.simulation import get_dataloader, load_pretrained_model
from awave.utils.warmstart import warm_start

parser = argparse.ArgumentParser(description='Simulation Example')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--wave', type=str, default='db5', help='type of wavelet')
parser.add_argument('--J', type=int, default=4, help='level of resolution')
parser.add_argument('--mode', type=str, default='zero', help='mode of wavelet boundary')
parser.add_argument('--init_factor', type=float, default=1, metavar='N', help='initialization parameter')
parser.add_argument('--noise_factor', type=float, default=0.3, metavar='N', help='initialization parameter')
parser.add_argument('--const_factor', type=float, default=0.0, metavar='N', help='initialization parameter')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--attr_methods', type=str, default='Saliency', help='type of attribution methods to penalize')
parser.add_argument('--lamlSum', type=float, default=1, help='weight of sum of lowpass filter')
parser.add_argument('--lamhSum', type=float, default=1, help='weight of sum of highpass filter')
parser.add_argument('--lamL2norm', type=float, default=1, help='weight of L2norm of lowpass filter')
parser.add_argument('--lamCMF', type=float, default=1, help='weight of CMF condition')
parser.add_argument('--lamConv', type=float, default=1, help='weight of convolution constraint')
parser.add_argument('--lamL1wave', type=float, default=0, help='weight of the l1-norm of wavelet coeffs')
parser.add_argument('--lamL1attr', type=float, default=0, help='weight of the l1-norm of attributions')
parser.add_argument('--target', type=int, default=0, help='target index to calculate interp score')
parser.add_argument('--dirname', default='dirname', help='name of directory')
parser.add_argument('--warm_start', default=None, help='indicate whether warmstart or not')


class p:
    """Parameters for simulated data
    """
    # data & model path
    data_path = "../../src/dsets/simulation/data"
    model_path = "../../src/dsets/simulation/data"
    wt_type = 'DWT1d'

    # parameters for generating data
    seed = 1
    n = 55000
    n_train = 50000
    n_test = 5000
    d = 64
    scale_knockout = 3
    idx_knockout = 8
    window = 1
    wave_gt = 'db5'  # ground truth wavelet
    noise_level = 0.1

    # parameters for wavelet initialization
    wave = 'db5'
    J = 4
    mode = 'zero'
    init_factor = 1
    noise_factor = 0.3
    const_factor = 0

    # parameters for training
    batch_size = 100
    lr = 0.001
    num_epochs = 50
    attr_methods = 'Saliency'
    lamlSum = 1
    lamhSum = 1
    lamL2norm = 1
    lamCMF = 1
    lamConv = 1
    lamL1wave = 0.1
    lamL1attr = 1
    target = 0

    # run with warmstart
    warm_start = None

    # SAVE MODEL
    out_dir = "/home/ubuntu/adaptive-wavelets/notebooks/simulation/results"
    dirname = "dirname"
    pid = ''.join(["%s" % random.randint(0, 9) for num in range(0, 10)])

    def _str(self):
        vals = vars(p)
        return 'wave=' + str(vals['wave']) + '_lamL1wave=' + str(vals['lamL1wave']) + '_lamL1attr=' + str(
            vals['lamL1attr']) \
               + '_seed=' + str(vals['seed']) + '_pid=' + vals['pid']

    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                if not attr.startswith('_')}


class s:
    '''Parameters to save
    '''

    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                if not attr.startswith('_')}


if __name__ == '__main__':
    args = parser.parse_args()
    for arg in vars(args):
        setattr(p, arg, getattr(args, arg))

    # create dir
    out_dir = opj(p.out_dir, p.dirname)
    os.makedirs(out_dir, exist_ok=True)

    # load data and model
    train_loader, test_loader = get_dataloader(p.data_path,
                                               batch_size=p.batch_size)

    model = load_pretrained_model(p.model_path)

    # prepare model
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    if p.warm_start is None:
        wt = DWT1d(wave=p.wave, mode=p.mode, J=p.J,
                   init_factor=p.init_factor,
                   noise_factor=p.noise_factor,
                   const_factor=p.const_factor).to(device)
        wt.train()
    else:
        wt = warm_start(p, out_dir).to(device)
        wt.train()

    # check if we have multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        wt = torch.nn.DataParallel(wt)

    # train
    params = list(wt.parameters())
    optimizer = torch.optim.Adam(params, lr=p.lr)
    loss_f = get_loss_f(lamlSum=p.lamlSum, lamhSum=p.lamhSum, lamL2norm=p.lamL2norm, lamCMF=p.lamCMF, lamConv=p.lamConv,
                        lamL1wave=p.lamL1wave, lamL1attr=p.lamL1attr)
    trainer = Trainer(model, wt, optimizer, loss_f, target=p.target,
                      use_residuals=True, attr_methods=p.attr_methods, device=device, n_print=5)

    # run
    trainer(train_loader, epochs=p.num_epochs)

    # calculate losses
    print('calculating losses and metric...')
    validator = Validator(model, test_loader)
    rec_loss, lsum_loss, hsum_loss, L2norm_loss, CMF_loss, conv_loss, L1wave_loss, L1saliency_loss, L1inputxgrad_loss = validator(
        wt, target=p.target)
    s.train_losses = trainer.train_losses
    s.rec_loss = rec_loss
    s.lsum_loss = lsum_loss
    s.hsum_loss = hsum_loss
    s.L2norm_loss = L2norm_loss
    s.CMF_loss = CMF_loss
    s.conv_loss = conv_loss
    s.L1wave_loss = L1wave_loss
    s.L1saliency_loss = L1saliency_loss
    s.L1inputxgrad_loss = L1inputxgrad_loss
    s.net = wt

    # save
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, p._str(p) + '.pkl'), 'wb'))
    if torch.cuda.device_count() > 1:
        torch.save(wt.module.state_dict(), opj(out_dir, p._str(p) + '.pth'))
    else:
        torch.save(wt.state_dict(), opj(out_dir, p._str(p) + '.pth'))
