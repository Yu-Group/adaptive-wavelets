import unittest

import numpy as np


class TestBasic(unittest.TestCase):

    def test_DWT1d(self):
        '''Test on synthetic dataset
        '''

        # params
        wave = 'db5'
        mode = 'zero'
        J = 4
        X = np.array([[0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert True
