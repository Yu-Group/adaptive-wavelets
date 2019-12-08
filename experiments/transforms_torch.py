import transforms_np
import torch
'''
im = torch.Tensor(mnu_dataset[0]['image'].astype(np.float32))
cshow(im)

# im_new = torch.zeros(list(im.shape) + [2]) # add imag dim
# im_new[:, :, 0] = im
# im_f = torch.fft(im_new, signal_ndim=2)
'''


def bandpass_filter(im: torch.Tensor, band_center=0.3, band_width=0.1, sample_spacing=None):
    '''Bandpass filter the image (assumes the image is square)
    
    Returns
    -------
    im_bandpass: torch.Tensor
    '''
    im_np = im.squeeze().cpu().detach().numpy()
    
    im_bandpass = transforms_np.bandpass_filter_norm_fast(im_np, 
                                                          cutoff_low=band_center - band_width / 2, 
                                                          cutoff_high=band_center + band_width / 2, 
                                                          kernel_length=25)
#     im_bandpass = transforms_np.bandpass_filter(im_np, band_center, band_width, sample_spacing)
    
    
    return torch.Tensor(im_bandpass).reshape(1, 1, im_np.shape[0], im_np.shape[1])

'''This code from https://github.com/tomrunia/PyTorchSteerablePyramid
'''

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)