import numpy as np

def to_freq(x):
    x =  x.cpu().detach().numpy().squeeze()
    return np.fft.fftshift(mag(x))


def mag(x):
    '''Magnitude
    x[..., 0] is real part
    x[..., 1] is imag part
    '''
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]))