import numpy as np
import torch
import torch.nn.functional as F

class PeakCount():
    """ Peak counting model."""
    
    def __init__(self, peak_counting_method='laplace_v1',
                 bins=np.linspace(-0.03,0.19,23),
                 kernels=None):
        self.peak_counting_method = peak_counting_method
        self.bins = bins
        self.kernels = kernels
        assert self.peak_counting_method in ['original',
                                             'laplace_v1',
                                             'laplace_v2',
                                             'roberts_cross',
                                             'custom']
        if self.peak_counting_method == 'custom':
            assert kernels is not None, "if using custom kernels, must also pass kernels!"
        
        
    def fit(self, dataloader):
        """Calculate the mean peak counts and covariances."""
        self.peak_list = {}
        for data, params in dataloader:
            n_batch = data.size(0)
            hps = self.peak_count(data)
            for i in range(n_batch):
                params_list = tuple([params[i,k].item() for k in range(3)])
                if params_list in self.peak_list:
                    self.peak_list[params_list].append(hps[i])
                else:
                    self.peak_list[params_list] = [hps[i]]

        self.mean_peaks = {}  
        self.inv_cov = {}   
        for k,v in self.peak_list.items():
            self.mean_peaks[k] = np.mean(v, axis=0)
            vm = np.vstack(v).T
            self.inv_cov[k] = np.linalg.pinv(np.cov(vm))    
                

    def predict(self, dataloader):
        """Predict on image with peak counts."""
        y_preds = []
        y_params = []
        ks = sorted(self.mean_peaks.keys())
        for data, params in dataloader:
            n_batch = data.size(0)
            hps = self.peak_count(data)
            
            chis = []
            for k in ks:
                d = hps - self.mean_peaks[k]
                chis.append((d.T * (self.inv_cov[k] @ d.T)).sum(axis=0))        
            chis = np.vstack(chis).T
            idx = np.argmin(chis, axis=1)
            
            preds = []
            for i in idx:
                preds.append(ks[i])
            preds = np.vstack(preds)  
            y_preds.append(preds)
            y_params.append(params.cpu().numpy())
        y_preds = np.vstack(y_preds)
        y_params = np.vstack(y_params)
        return y_preds, y_params   


    def find_peaks(self, ims):
        """Find peaks in bw image."""
        assert len(ims.shape)==4, "shape = (batch size, n_channel, height, width)"
        p =  ims[...,1:-1,1:-1] > ims[...,:-2,:-2]  # top left
        p &= ims[...,1:-1,1:-1] > ims[...,:-2,1:-1]  # top center  
        p &= ims[...,1:-1,1:-1] > ims[...,:-2,2:]  # top right
        p &= ims[...,1:-1,1:-1] > ims[...,1:-1,:-2]  # center left 
        p &= ims[...,1:-1,1:-1] > ims[...,1:-1,2:]  # center right 
        p &= ims[...,1:-1,1:-1] > ims[...,2:,:-2]  # bottom left
        p &= ims[...,1:-1,1:-1] > ims[...,2:,1:-1]  # bottom center
        p &= ims[...,1:-1,1:-1] > ims[...,2:,2:]   # bottom right
        return p
    
    
    def images_at_peaks(self, ims_f, p):
        """Get filtered images at the position of peaks"""
        n_batch = p.size(0)
        vals = ims_f[p]    
        n_peaks = p.sum(axis=(1,2,3))
        n_peaks = n_peaks.cpu().numpy()
        idx = list(np.cumsum(n_peaks))
        idx.insert(0,0) # index starts from 0
        results = []
        for i in range(n_batch):
            left, right = idx[i], idx[i+1]
            vals_b = vals[left:right]
            results.append(vals_b.cpu().numpy())    
        return results    
    

    def peak_count(self, ims):
        """Peak counting statistics"""
        peaks = self.find_peaks(ims)  # find peaks  

        # get the values for peaks
        if self.peak_counting_method == 'custom':
            hps = []
            for kernel in self.kernels:
                kernel = kernel[None,None]
                ims_f = F.conv2d(ims, kernel)
                vals = self.images_at_peaks(ims_f, peaks)

                # make histogram
                results = []
                for i,val in enumerate(vals):
                    hp = np.histogram(val, bins=self.bins)[0]
                    results.append(hp) 
                hps.append(np.vstack(results))
            return np.hstack(hps)              
        else:
            if self.peak_counting_method == 'original':
                ims_f = ims[...,1:-1,1:-1]
            elif self.peak_counting_method == 'laplace_v1':
                ims_f = self.laplace_v1(ims)
            elif self.peak_counting_method == 'laplace_v2':
                ims_f = self.laplace_v2(ims)
            elif self.peak_counting_method == 'roberts_cross':
                ims_f = self.roberts_cross(ims)
            vals = self.images_at_peaks(ims_f, peaks)
            # make histogram
            results = []
            for i,val in enumerate(vals):
                hp = np.histogram(val, bins=self.bins)[0]
                results.append(hp)    
            return np.vstack(results)        
    

    def laplace_v1(self, ims):
        """Characterize peaks with laplace kernel in image."""
        L = 10/3 * torch.tensor([[-0.05, -0.2, -0.05], 
                                 [-0.2, 1.0, -0.2], 
                                 [-0.05, -0.2, -0.05]])  
        L = L[None,None]
        ims_f = F.conv2d(ims, L)
        return ims_f
    

    def laplace_v2(self, ims):
        """Characterize peaks with laplace kernel in image."""
        L = 4 * torch.tensor([[0, -0.25, 0], 
                              [-0.25, 1.0, -0.25], 
                              [0, -0.25, 0]])   
        L = L[None,None]
        ims_f = F.conv2d(ims, L)
        return ims_f
    

    def roberts_cross(self, ims):
        """Evaluate Robert's cross gradient magnitude."""
        Rx = torch.tensor([[0.0, 1.0], 
                           [-1.0, 0.0]])
        Ry = torch.tensor([[1.0, 0.0], 
                           [0.0, -1.0]])  
        Rx = Rx[None,None]
        Ry = Ry[None,None]
        p0 = F.conv2d(ims[...,:-1,:-1], Rx)**2
        p0 += F.conv2d(ims[...,:-1,:-1], Ry)**2
        p0 = torch.sqrt(p0)
        p1 = F.conv2d(ims[...,:-1,1:], Rx)**2
        p1 += F.conv2d(ims[...,:-1,1:], Ry)**2
        p1 = torch.sqrt(p1)
        p2 = F.conv2d(ims[...,1:,:-1], Rx)**2
        p2 += F.conv2d(ims[...,1:,:-1], Ry)**2
        p2 = torch.sqrt(p2)    
        p3 = F.conv2d(ims[...,1:,1:], Rx)**2
        p3 += F.conv2d(ims[...,1:,1:], Ry)**2
        p3 = torch.sqrt(p3)    
        return (p0+p1+p2+p3)
    
    
    def center_kernels(self):
        for _ in range(len(self.kernels)):
            kern = self.kernels.pop(0)
            kern = kern - kern.mean()
            self.kernels.append(kern)
            
            

class ModelPred():
    """ Predict cosmological parameters using neural network model."""
    
    def __init__(self, model, target=1):
        self.model = model
        self.device = model.fc.weight.device
        self.target = target

    def fit(self, dataloader):
        """Calculate the mean outputs and covariances."""
        self.output_list = {}
        with torch.no_grad():
            for data, params in dataloader:
                data = data.to(self.device)
                n_batch = data.size(0)
                outputs = self.model(data).detach().cpu().numpy()
                for i in range(n_batch):
                    params_list = tuple([params[i,k].item() for k in range(3)])
                    if params_list in self.output_list:
                        self.output_list[params_list].append(outputs[i])
                    else:
                        self.output_list[params_list] = [outputs[i]]
                        
        self.mean_outputs = {}  
        for k,v in self.output_list.items():
            self.mean_outputs[k] = np.mean(v, axis=0)

            
    def predict(self, dataloader):
        """Predict on image with model outputs."""
        y_preds = []
        y_params = []
        ks = sorted(self.mean_outputs.keys())
        with torch.no_grad():
            for data, params in dataloader:
                data = data.to(self.device)
                n_batch = data.size(0)
                outputs = self.model(data).detach().cpu().numpy()

                chis = []
                for k in ks:
                    d = outputs[:,self.target] - self.mean_outputs[k][self.target]
                    chis.append(d**2)
                chis = np.vstack(chis).T
                idx = np.argmin(chis, axis=1)

                preds = []
                for i in idx:
                    preds.append(ks[i])
                preds = np.vstack(preds)  
                y_preds.append(preds)
                y_params.append(params.cpu().numpy())
        y_preds = np.vstack(y_preds)
        y_params = np.vstack(y_params)
        return y_preds, y_params  
    
            
            
def rmse(y_params, y_preds, target=1):
    return np.linalg.norm(y_params[:,target] - y_preds[:,target])**2/y_params.shape[0]
       
    
    