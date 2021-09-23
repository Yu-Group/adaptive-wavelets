import os

import numpy as np
import torch

opj = os.path.join
import pickle as pkl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from awave.transform1d import DWT1d
from awave.transform2d import DWT2d


def warm_start(p, out_dir):
    '''load results and initialize model 
    '''
    print('\twarm starting...')
    fnames = sorted(os.listdir(out_dir))
    lamL1attr = []
    lamL1wave = []
    models = []
    if len(fnames) == 0:
        if p.wt_type == 'DWT1d':
            model = DWT1d(wave=p.wave, mode=p.mode, J=p.J, init_factor=p.init_factor, noise_factor=p.noise_factor).to(
                device)
        elif p.wt_type == 'DWT2d':
            model = DWT2d(wave=p.wave, mode=p.mode, J=p.J, init_factor=p.init_factor, noise_factor=p.noise_factor).to(
                device)
    else:
        for fname in fnames:
            if fname[-3:] == 'pkl':
                result = pkl.load(open(opj(out_dir, fname), 'rb'))
                lamL1attr.append(result['lamL1attr'])
                lamL1wave.append(result['lamL1wave'])
            if fname[-3:] == 'pth':
                if p.wt_type == 'DWT1d':
                    m = DWT1d(wave=p.wave, mode=p.mode, J=p.J, init_factor=p.init_factor,
                              noise_factor=p.noise_factor).to(device)
                elif p.wt_type == 'DWT2d':
                    m = DWT2d(wave=p.wave, mode=p.mode, J=p.J, init_factor=p.init_factor,
                              noise_factor=p.noise_factor).to(device)
                m.load_state_dict(torch.load(opj(out_dir, fname)))
                models.append(m)
        lamL1attr = np.array(lamL1attr)
        lamL1wave = np.array(lamL1wave)
        if p.lamL1attr == 0:
            lamL1wave_max = np.max(lamL1wave[lamL1attr == 0])
            idx = np.argwhere((lamL1attr == 0) & (lamL1wave == lamL1wave_max)).item()
        else:
            lamL1attr_max = np.max(lamL1attr[lamL1wave == p.lamL1wave])
            idx = np.argwhere((lamL1attr == lamL1attr_max) & (lamL1wave == p.lamL1wave)).item()
        model = models[idx]
        print('initialized at the model with lamL1wave={:.5f} and lamL1attr={:.5f}'.format(lamL1wave[idx],
                                                                                           lamL1attr[idx]))
    return model
