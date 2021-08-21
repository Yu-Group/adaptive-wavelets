import unittest

import numpy as np

from awd.transform1d import DWT1d
from awd.utils.misc import get_wavefun


class TestBasic(unittest.TestCase):

    def test_DWT1d(self):
        '''Test on synthetic dataset
        '''

        # params
        wave = 'db5'
        mode = 'zero'
        device = 'cpu'
        J = 4

        # initialize model
        wt = DWT1d(wave=wave, mode=mode, J=J).to(device)

        # visualize
        phi_orig, psi_orig, x_orig = get_wavefun(wt)

        # some random data
        X = np.random.randn(3, 1, 6)  # batch_size,

        # fit the random data
        wt.fit(X=X, lr=1e-2, num_epochs=1)  # this function alternatively accepts a dataloader

        assert True
