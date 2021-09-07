import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch.autograd import Function

from awave.utils.misc import reflect

def load_wavelet(wave: str, device=None):
    '''load wavelet from pywt (currently only allow orthogonal wavelets)
    '''
    wave = pywt.Wavelet(wave)
    h0, h1 = wave.dec_lo, wave.dec_hi
    g0, g1 = wave.rec_lo, wave.rec_hi
    # Prepare the filters
    h0, h1 = prep_filt_afb1d(h0, h1, device)
    g0, g1 = prep_filt_sfb1d(g0, g1, device)
    if not torch.allclose(h0, g0) or not torch.allclose(h1, g1):
        raise ValueError('currently only orthogonal wavelets are supported')
    return h0, h1

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n + end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:, -n:], x[:, :-n + end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:, :, -n:], x[:, :, :-n + end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:, :, :, -n:], x[:, :, :, :-n + end]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, :, xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1 + m2, dtype='int32'), -0.5, l1 - 0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2 + m2, dtype='int32'), -0.5, l2 - 0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:, :, xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:, :, :, xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            else:
                x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:, :, :L2] = lohi[:, :, :L2] + lohi[:, :, N2:N2 + L2]
            lohi = lohi[:, :, :N2]
        else:
            lohi[:, :, :, :L2] = lohi[:, :, :, :L2] + lohi[:, :, :, N2:N2 + L2]
            lohi = lohi[:, :, :, :N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi


def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:, :, :L - 2] = y[:, :, :L - 2] + y[:, :, N:N + L - 2]
            y = y[:, :, :N]
        else:
            y[:, :, :, :L - 2] = y[:, :, :, :L - 2] + y[:, :, :, N:N + L - 2]
            y = y[:, :, :, :N]
        y = roll(y, 1 - L // 2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


class AFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """

    @staticmethod
    def forward(x, h0_row, h1_row, h0_col, h1_col, mode):
        mode = int_to_mode(mode)
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs


class AFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """

    @staticmethod
    def forward(x, h0, h1, mode):
        mode = int_to_mode(mode)

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        lohi = afb1d(x, h0, h1, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1


class SFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """

    @staticmethod
    def forward(low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        mode = int_to_mode(mode)

        lh, hl, hh = torch.unbind(highs, dim=2)
        lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y


class SFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        low (torch.Tensor): Lowpass to reconstruct of shape (N, C, L)
        high (torch.Tensor): Highpass to reconstruct of shape (N, C, L)
        g0: lowpass
        g1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*2, L')
    """

    @staticmethod
    def forward(low, high, g0, g1, mode):
        mode = int_to_mode(mode)
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        return sfb1d(low, high, g0, g1, mode=mode, dim=3)[:, :, 0]


def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1


def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row


def prep_filt_afb1d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1
