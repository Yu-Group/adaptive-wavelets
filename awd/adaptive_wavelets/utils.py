import numpy as np
import pywt
import torch
import torch.nn.functional as F


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def tuple_dim(x):
    tot_dim = 0
    for i in range(len(x)):
        shape = torch.tensor(x[i].shape)
        tot_dim += torch.prod(shape).item()
    return tot_dim


def tuple_to_tensor(x):
    batch_size = x[0].size(0)
    J = len(x)
    y = torch.tensor([]).to(x[0].device)
    list_of_size = [0]
    for j in range(J):
        a = x[j].reshape(batch_size, -1)
        y = torch.cat((y, a), dim=1)
        list_of_size.append(list_of_size[-1] + a.shape[1])
    return (y, list_of_size)


def tensor_to_tuple(y, d, list_of_size):
    x = []
    J = len(list_of_size) - 1
    for j in range(J):
        n0 = list_of_size[j]
        n1 = list_of_size[j + 1]
        x.append(y[:, n0:n1].reshape(d[j].shape))
    return tuple(x)


def init_filter(x, init_factor, noise_factor, const_factor):
    '''add random noise to tensor
    Params
    ------
    x: torch.tensor
        input
    init_factor: float

    noise_factor: float
        amount of noise added to original filter
        
    const_factor: float
        amount of constant added to original filter
    '''
    shape = x.shape
    x = init_factor * x + noise_factor * torch.randn(shape) + const_factor * torch.ones(shape)
    return x


def pad_within(x, stride=2, start_row=0, start_col=0):
    w = x.new_zeros(stride, stride)
    if start_row == 0 and start_col == 0:
        w[0, 0] = 1
    elif start_row == 0 and start_col == 1:
        w[0, 1] = 1
    elif start_row == 1 and start_col == 0:
        w[1, 0] = 1
    else:
        w[1, 1] = 1
    if len(x.shape) == 2:
        x = x[None, None]
    return F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1)).squeeze()


def low_to_high(x):
    """Converts lowpass filter to highpass filter. Input must be of shape (1,1,n) where n is length of filter
    """
    n = x.size(2)
    seq = (-1) ** torch.arange(n, device=x.device)
    y = torch.flip(x, (0, 2)) * seq
    return y


def get_wavefun(w_transform, level=5):
    '''Get wavelet function from wavelet object.
    Params
    ------
    w_transform: obj
        DWT1d or DWT2d object
    '''
    h0 = w_transform.h0
    h1 = low_to_high(h0)

    h0 = list(h0.squeeze().detach().cpu().numpy())[::-1]
    h1 = list(h1.squeeze().detach().cpu().numpy())[::-1]

    my_filter_bank = (h0, h1, h0[::-1], h1[::-1])
    my_wavelet = pywt.Wavelet('My Wavelet', filter_bank=my_filter_bank)
    wave = my_wavelet.wavefun(level=level)
    (phi, psi, x) = wave[0], wave[1], wave[4]

    return phi, psi, x


def dist(wt1, wt2):
    """function to compute distance between two wavelets 
    """
    _, psi1, _ = get_wavefun(wt1)
    _, psi2, _ = get_wavefun(wt2)

    if len(psi1) > len(psi2):
        psi2 = np.pad(psi2, (0, len(psi1) - len(psi2)), mode='constant', constant_values=(0,))
    if len(psi1) < len(psi2):
        psi1 = np.pad(psi1, (0, len(psi2) - len(psi1)), mode='constant', constant_values=(0,))

    distance = []
    # circular shift 
    for i in range(len(psi1)):
        psi1_r = np.roll(psi1, i)
        d = np.linalg.norm(psi1_r - psi2)
        distance.append(d.item())
    # flip filter
    psi1_f = psi1[::-1]
    for i in range(len(psi1)):
        psi1_r = np.roll(psi1_f, i)
        d = np.linalg.norm(psi1_r - psi2)
        distance.append(d.item())

    return min(distance)


def get_1dfilts(w_transform):
    '''Get 1d filters from DWT1d object.
    Params
    ------
    w_transform: obj
        DWT1d object
    '''
    if w_transform.wt_type == 'DWT1d':
        h0 = w_transform.h0.squeeze().detach().cpu()
        h1 = low_to_high(w_transform.h0)
        h1 = h1.squeeze().detach().cpu()
        h0 = F.pad(h0, pad=(0, 0), mode='constant', value=0)
        h1 = F.pad(h1, pad=(0, 0), mode='constant', value=0)
        return (h0, h1)
    else:
        raise ValueError('no such type of wavelet transform is supported')


def get_2dfilts(w_transform):
    '''Get 2d filters from DWT2d object.
    Params
    ------
    w_transform: obj
        DWT2d object
    '''
    if w_transform.wt_type == 'DTCWT2d':
        h0o = w_transform.xfm.h0o.data
        h1o = w_transform.xfm.h1o.data
        h0a = w_transform.xfm.h0a.data
        h1a = w_transform.xfm.h1a.data
        h0b = w_transform.xfm.h0b.data
        h1b = w_transform.xfm.h1b.data

        # compute first level wavelet filters
        h0_r = F.pad(h0o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h0_i = F.pad(h0o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)
        h1_r = F.pad(h1o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h1_i = F.pad(h1o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)

        lh_filt_r1 = h0_r.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        lh_filt_r2 = h0_i.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        lh_filt_i1 = h0_i.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        lh_filt_i2 = h0_r.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = h1_r.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        hh_filt_r2 = h1_i.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        hh_filt_i1 = h1_i.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        hh_filt_i2 = h1_r.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2

        hl_filt_r1 = h1_r.unsqueeze(0) * h0_r.unsqueeze(1) / np.sqrt(2)
        hl_filt_r2 = h1_i.unsqueeze(0) * h0_i.unsqueeze(1) / np.sqrt(2)
        hl_filt_i1 = h1_i.unsqueeze(0) * h0_r.unsqueeze(1) / np.sqrt(2)
        hl_filt_i2 = h1_r.unsqueeze(0) * h0_i.unsqueeze(1) / np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2

        fl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        fl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]

        # compute second level wavelet filters
        h0_a = h0a.squeeze().detach().cpu()
        h0_b = h0b.squeeze().detach().cpu()
        h1_a = h1a.squeeze().detach().cpu()
        h1_b = h1b.squeeze().detach().cpu()

        lh_filt_r1 = pad_within(h0_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        lh_filt_r2 = pad_within(h0_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        lh_filt_i1 = pad_within(h0_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        lh_filt_i2 = pad_within(h0_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = pad_within(h1_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        hh_filt_r2 = pad_within(h1_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        hh_filt_i1 = pad_within(h1_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        hh_filt_i2 = pad_within(h1_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2

        hl_filt_r1 = pad_within(h1_a.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        hl_filt_r2 = pad_within(h1_b.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        hl_filt_i1 = pad_within(h1_b.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        hl_filt_i2 = pad_within(h1_a.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2

        sl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        sl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]

        return (fl_filt_reals, fl_filt_imags), (sl_filt_reals, sl_filt_imags)

    elif w_transform.wt_type == 'DWT2d':
        h0 = w_transform.h0.squeeze().detach().cpu()
        h1 = low_to_high(w_transform.h0)
        h1 = h1.squeeze().detach().cpu()
        h0 = F.pad(h0, pad=(0, 0), mode='constant', value=0)
        h1 = F.pad(h1, pad=(0, 0), mode='constant', value=0)

        filt_ll = h0.unsqueeze(0) * h0.unsqueeze(1)
        filt_lh = h0.unsqueeze(0) * h1.unsqueeze(1)
        filt_hl = h1.unsqueeze(0) * h0.unsqueeze(1)
        filt_hh = h1.unsqueeze(0) * h1.unsqueeze(1)

        return (h0, h1), (filt_ll, filt_lh, filt_hl, filt_hh)

    else:
        raise ValueError('no such type of wavelet transform is supported')
