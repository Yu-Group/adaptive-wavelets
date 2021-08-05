import numpy as np
import scipy.signal as signal
from numpy.fft import *


def bandpass_filter(im: np.ndarray, band_center=0.3, band_width=0.1, sample_spacing=None, mask=None):
    '''Bandpass filter the image (assumes the image is square)
    
    Returns
    -------
    im_bandpass: np.ndarray
    mask: np.ndarray
        if mask is present, use this mask to set things to 1 instead of bandpass
    '''

    # find freqs
    if sample_spacing is None:  # use normalized freqs [-1, 1]
        freq_arr = fftshift(fftfreq(n=im.shape[0]))
        freq_arr /= np.max(np.abs(freq_arr))
    else:
        sample_spacing = 0.8  # arcmins
        freq_arr = fftshift(fftfreq(n=im.shape[0], d=sample_spacing))  # 1 / arcmin
        # print(freq_arr[0], freq_arr[-1])

    # go to freq domain
    im_f = fftshift(fft2(im))
    '''
    plt.imshow(np.abs(im_f))
    plt.xlabel('frequency x')
    plt.ylabel('frequency y')
    '''

    # bandpass
    if mask is not None:
        assert mask.shape == im_f.shape, 'mask shape does not match shape in freq domain'
        mask_bandpass = mask
    else:
        mask_bandpass = np.zeros(im_f.shape)
        for r in range(im_f.shape[0]):
            for c in range(im_f.shape[1]):
                dist = np.sqrt(freq_arr[r] ** 2 + freq_arr[c] ** 2)
                if dist > band_center - band_width / 2 and dist < band_center + band_width / 2:
                    mask_bandpass[r, c] = 1

    im_f_masked = np.multiply(im_f, mask_bandpass)
    '''
    plt.imshow(np.abs(im_f_masked))
    plt.xticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
    plt.yticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
    plt.show()
    '''

    im_bandpass = np.real(ifft2(ifftshift(im_f_masked)))
    return im_bandpass


'''
Written by Alan Dong
Based on Jae S. Lim, "Two Dimensional Signal and Image Processing" 1990
'''


def bandpass_filter_norm_fast(im: np.ndarray, cutoff_low=0.25, cutoff_high=0.75, kernel_length=25):
    '''Return bandpass-filtered image, with freqs normalized
    '''

    def ftrans2(b: np.ndarray, t=None):
        '''Implements McClellan transform which produces 2D filter from 1D filter

        Params
        ------
        b - 1D filter        
        t - transform matrix, defaults to McClellan transformation
        '''
        if len(b.squeeze().shape) > 1:
            raise Exception("ftrans2: b must be a one dimensional array!")
        elif np.all(b == 0):
            raise Exception("ftrans2: b must have at least one nonzero element!")
        elif len(b) % 2 == 0:
            raise Exception("ftrans2: b must be odd length!")
        elif np.any(abs(b - b[::-1]) > np.sqrt(np.finfo(b.dtype).eps)):
            raise Exception("ftrans2: b must be symmetric!")

        if t is None:
            t = np.array([[1., 2, 1], [2, -4, 2], [1, 2, 1]]) / 8.  # McClellan transformation
        n = (len(b) - 1) // 2
        b = np.fft.ifftshift(b)
        a = np.concatenate([[b[0]], 2.0 * b[1:n + 1]])

        inset = np.floor((np.array(t.shape) - 1) / 2).astype("int")

        # Use Chebyshev polynomials to compute h
        P0 = 1
        P1 = t
        h = a[1] * P1
        rows = np.array([inset[0]])
        cols = np.array([inset[1]])
        h[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = h[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] + a[0] * P0
        for i in range(2, n + 1):
            P2 = 2 * signal.convolve2d(t, P1)
            rows = rows + inset[0]
            cols = cols + inset[1]
            P2[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = P2[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] - P0
            rows = inset[0] + np.arange(P1.shape[0])
            cols = inset[1] + np.arange(P1.shape[1])
            hh = h
            h = a[i] * P2
            h[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = h[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] + hh
            P0 = P1
            P1 = P2
        return h

    def filter2(im: np.ndarray, h: np.ndarray):
        '''2D filtering
        Params
        ------
        im - image to be filtered
        h - 2D filter
        '''
        if np.issubdtype(im.dtype, np.integer):
            im = im.astype("float")
        if len(im.shape) == 2:
            out = signal.convolve2d(im, h, "same")
        elif len(im.shape) == 3:
            out = np.zeros(im.shape, dtype=im.dtype)
            for i in range(im.shape[2]):
                out[..., i] = signal.convolve2d(im[..., i], h, "same")
        else:
            raise Exception("filter2: im must be two or three dimensional!")
        return out

    b = signal.firwin(kernel_length, cutoff=[cutoff_low, cutoff_high], window='blackmanharris', pass_zero=False)
    h = ftrans2(b)
    return filter2(im, h)
