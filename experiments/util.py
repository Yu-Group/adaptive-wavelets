import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../dsets/mnist')
import dset
from model import Net2c
from transforms_torch import bandpass_filter_augment

def to_freq(x):
    x =  x.cpu().detach().numpy().squeeze()
    return np.fft.fftshift(mag(x))


def mag(x):
    '''Magnitude
    x[..., 0] is real part
    x[..., 1] is imag part
    '''
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]))


# train Net2c for classifying original images vs filtered images
def train_Net2c(band_center=0.3, band_width=0.1, save_path='mnist/models/cnn_vanilla.pth'):
    # load mnist data
    args = dset.get_args()
    args.batch_size, args.test_batch_size = int(args.batch_size/2), int(args.test_batch_size/2) # half the batchsize
    args.epochs = 5 # set number of epochs to be 5
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = dset.load_data(args.batch_size, args.test_batch_size, args.cuda)

    # create model
    model = Net2c()
    if args.cuda:
        model.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # train
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = bandpass_filter_augment(data, band_center, band_width)
            target = torch.zeros(2*args.batch_size, dtype=target.dtype)
            target[args.batch_size:] = 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 2*len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()), end='')

    torch.save(model.state_dict(), save_path)
