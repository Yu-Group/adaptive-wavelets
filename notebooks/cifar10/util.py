import torch
from os.path import join as oj
import os

def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')

def test_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
    
def train(model, device, train_loader, test_loader, optimizer, num_epochs, criterion, save_dir=None):
    print('training...')
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    best_loss = 1e10
    test_losses = []
    for epoch in range(num_epochs):
        train_epoch(model, device, train_loader, optimizer, epoch+1, criterion)
        test_loss = test_epoch(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        
        # saving
        if test_loss < best_loss:
            best_loss = test_loss
            if save_dir is not None:
                torch.save(model.state_dict(),
                           oj(save_dir, f'checkpoint_{epoch}.pth'))

            