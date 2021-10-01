'''Code here adapted from https://distill.pub/2020/circuits/curve-circuits/
'''

import numpy as np

from awave.experimental.util import weight_axes


def gabor_filter(filter_size: int, angle: float, shift=0):
    '''
    Params
    ------
    filter_size
    angle: float
        in degrees

    Returns
    -------
    gabor_filter: np.ndarray
        shape size x size
    '''
    X, Y = weight_axes(filter_size)
    mask = 3 / (3 + ((X - filter_size // 2) ** 2 + (Y - filter_size // 2) ** 2) / 2.)
    angle *= 6.283 / 360  # convert to radians
    X_ = np.cos(angle) * X + np.sin(angle) * Y
    k = 4 * np.pi / filter_size
    X_ = X_ - filter_size // 2 + shift
    F = mask * np.cos(k * X_)
    return F / np.abs(F).max()


def edge_filter(filter_size: int, angle: float):
    '''
    Params
    ------
    filter_size
    angle: float
        in degrees

    Returns
    -------
    edge_filter: np.ndarray
        shape size x size
    edge_filter_surround: np.ndarray
        shape size x size
    '''
    X, Y = weight_axes(filter_size)
    mask = 3 / (3 + np.sqrt((X - filter_size // 2) ** 2 + (Y - filter_size // 2) ** 2) / 2.)
    angle *= 6.283 / 360  # convert to radians
    X, Y = X - filter_size // 2, Y - filter_size // 2
    X_ = np.cos(angle) * X + np.sin(angle) * Y
    k = 4 * np.pi / filter_size
    F = np.clip(1.1 - np.abs(X_), 0, 1)
    F /= np.maximum(np.abs(F).max(), 1e-6)
    return F, mask * (1 - F)


def curve_filter(filter_size: int, angle: float, radius: float):
    '''
    Params
    ------
    filter_size
    angle: float
       in degrees
    '''
    X, Y = weight_axes(filter_size)
    X, Y = X - filter_size // 2, Y - filter_size // 2
    angle *= 6.283 / 360  # convert to radians
    cx, cy = -radius * np.cos(angle), -radius * np.sin(angle)
    dx, dy = X - cx, Y - cy
    E = 1.05 - np.abs(np.sqrt(dx ** 2 + dy ** 2) - radius) / filter_size / 1.3
    E = np.clip(E, 0, 1)
    E = E ** 4
    E = 2 * E - 1
    # E = E**2
    E = np.clip(E, 0, 1)
    tanget_angles = np.arctan2(dy, dx) * 360 / 6.28 % 360

    return E, tanget_angles
