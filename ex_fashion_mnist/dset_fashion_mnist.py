from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


# Training settings
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args("")


# load data
def load_data(train_batch_size,
              test_batch_size,
              device,
              data_dir='data',
              shuffle=False,
              return_indices=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    FashionMNIST_dataset = datasets.FashionMNIST
    if return_indices:
        FashionMNIST_dataset = dataset_with_indices(FashionMNIST_dataset)

    train_loader = torch.utils.data.DataLoader(
        FashionMNIST_dataset(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])), batch_size=train_batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        FashionMNIST_dataset(data_dir, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])), batch_size=test_batch_size, shuffle=shuffle, **kwargs)
    return train_loader, test_loader


def train(epoch, train_loader, model, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()), end='')
    return model


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_im_and_label(num, device='cuda'):
    torch.manual_seed(130)
    _, data_loader = load_data(train_batch_size=1, test_batch_size=1,
                               device=device, data_dir='mnist/data',
                               shuffle=False)
    for i, im in enumerate(data_loader):
        if i == num:
            return im[0].to(device), im[0].numpy().squeeze(), im[1].numpy()[0]


def pred_ims(model, ims, layer='softmax', device='cuda'):
    if len(ims.shape) == 2:
        ims = np.expand_dims(ims, 0)
    ims_torch = torch.unsqueeze(torch.Tensor(ims), 1).float().to(device) # cuda()
    preds = model(ims_torch)

    # todo - build in logit support
    # logits = model.logits(t)
    return preds.data.cpu().numpy()


# mnist dataset with return index
# dataset
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


# dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, return_indices=True):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.return_indices = return_indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index].reshape(28,28), int(self.targets[index])
        if self.transform:
            img = self.transform(img)
        if self.return_indices:
            return img, target, index
        return img, target


def load_mnist_arrays(train_loader, test_loader):
    # dataset
    X = train_loader.dataset.data.numpy().astype(np.float32)
    X = X.reshape(X.shape[0], -1)
    X /= 255
    Y = train_loader.dataset.targets.numpy()

    X_test = test_loader.dataset.data.numpy().astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test /= 255
    Y_test = test_loader.dataset.targets.numpy()
    data_dict = {
        'data': X,
        'data_t': X_test,
        'targets': Y,
        'targets_t': Y_test,
    }
    return data_dict



if __name__ == '__main__':
    from model import Net
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.batch_size, args.test_batch_size, args.cuda)

    # create model
    model = Net()
    if args.cuda:
        model.cuda()

    # train
    for epoch in range(1, args.epochs + 1):
        model = train(epoch, train_loader)
        test(model, test_loader)

    # save
    torch.save(model.state_dict(), 'mnist.model')
    # load and test
    # model_loaded = Net().cuda()
    # model_loaded.load_state_dict(torch.load('mnist.model'))
    # test(model_loaded, test_loader)
