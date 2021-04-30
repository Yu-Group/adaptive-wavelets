import numpy as np
import torch
import random
from copy import deepcopy


def ftr_transform(w_transform, train_loader, test_loader):
    w_transform = w_transform.to('cpu')
    J = w_transform.J
    X = []
    y = []
    for data, labels in train_loader:
        data_t = w_transform(data)
        for j in range(J+1):
            if j == 0:
                x = deepcopy(data_t[j].detach()).squeeze()
            else:
                a = deepcopy(torch.max(data_t[j].detach(), dim=2)[0])
                b = deepcopy(-torch.max(-data_t[j].detach(), dim=2)[0])
                f = -2*torch.max(torch.cat((a,-b),axis=1),dim=1)[1] + 1
                x1 = torch.max(torch.cat((a,-b),axis=1),dim=1)[0] * f                        
                x = torch.cat((x,x1[:,None]), axis=1)  
        X.append(x)
        y.append(labels)
    X = torch.cat(X).squeeze().numpy()
    y = torch.cat(y).squeeze().numpy()

    X_test = []
    y_test = []
    for data, labels in test_loader:
        data_t = w_transform(data)
        for j in range(J+1):
            if j == 0:
                x = deepcopy(data_t[j].detach()).squeeze()
            else:
                a = deepcopy(torch.max(data_t[j].detach(), dim=2)[0])
                b = deepcopy(-torch.max(-data_t[j].detach(), dim=2)[0])
                f = -2*torch.max(torch.cat((a,-b),axis=1),dim=1)[1] + 1
                x1 = torch.max(torch.cat((a,-b),axis=1),dim=1)[0] * f                        
                x = torch.cat((x,x1[:,None]), axis=1)            
        X_test.append(x)
        y_test.append(labels)
    X_test = torch.cat(X_test).squeeze().numpy()
    y_test = torch.cat(y_test).squeeze().numpy()   
    
    return (X, y), (X_test, y_test)


def max_ftr_transform(w_transform, train_loader, test_loader):
    w_transform = w_transform.to('cpu')
    J = w_transform.J
    X = []
    y = []
    for data, labels in train_loader:
        data_t = w_transform(data)
        for j in range(J+1):
            a = deepcopy(torch.max(data_t[j].detach(), dim=2)[0])
            b = deepcopy(-torch.max(-data_t[j].detach(), dim=2)[0])
            f = -2*torch.max(torch.cat((a,-b),axis=1),dim=1)[1] + 1
            x1 = torch.max(torch.cat((a,-b),axis=1),dim=1)[0] * f                        
            if j == 0:
                x = x1[:,None]
            else:
                x = torch.cat((x,x1[:,None]), axis=1)
        X.append(x)
        y.append(labels)
    X = torch.cat(X).squeeze().numpy()
    y = torch.cat(y).squeeze().numpy()

    X_test = []
    y_test = []
    for data, labels in test_loader:
        data_t = w_transform(data)
        for j in range(J+1):
            a = deepcopy(torch.max(data_t[j].detach(), dim=2)[0])
            b = deepcopy(-torch.max(-data_t[j].detach(), dim=2)[0])
            f = -2*torch.max(torch.cat((a,-b),axis=1),dim=1)[1] + 1
            x1 = torch.max(torch.cat((a,-b),axis=1),dim=1)[0] * f            
            if j == 0:
                x = x1[:,None]
            else:
                x = torch.cat((x,x1[:,None]), axis=1)
        X_test.append(x)
        y_test.append(labels)
    X_test = torch.cat(X_test).squeeze().numpy()
    y_test = torch.cat(y_test).squeeze().numpy()   
    
    return (X, y), (X_test, y_test)    


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=500000, sparsity=None, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __hard_thresh(self):
        index = np.argsort(abs(self.theta[1:]))
        self.theta[1:][index[:-self.sparsity]] = 0
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        # store loss
        self.train_losses = []
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            if self.sparsity is not None:
                self.__hard_thresh()
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            self.train_losses.append(loss)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
            
            if len(self.train_losses) >= 2 and self.train_losses[-2] - self.train_losses[-1] < 1e-12:
                print('convergence criterion reached')
                break
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
    