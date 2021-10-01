'''Code here adapted from https://distill.pub/2020/circuits/curve-circuits/
'''

import numpy as np


def pn_image(x):
    p, n = np.maximum(0, x), np.maximum(0, -x)
    return np.stack([p, (p + n) / 2., n], axis=-1)


def weight_axes(N):
    '''
    Ex. weight_axes(3)
    Returns (array([[0, 1, 2]]), array([[0], [1], [2]])) # (1, 3) and (3, 1)
    '''
    X = np.arange(N)
    return X[None, :], X[:, None]


H = lambda S: int(S, 16) / 255.
C = lambda X: np.asarray([H(X[0:2]), H(X[2:4]), H(X[4:6])])


def weight_color_scale(x):
    if x < 0:
        x = -x
        if x < 0.5:
            x = x * 2
            return (1 - x) * C("f7f7f7") + x * C("92c5de")
        else:
            x = (x - 0.5) * 2
            return (1 - x) * C("92c5de") + x * C("0571b0")
    else:
        if x < 0.5:
            x = x * 2
            return (1 - x) * C("f7f7f7") + x * C("f4a582")
        else:
            x = (x - 0.5) * 2
            return (1 - x) * C("f4a582") + x * C("ca0020")


def weight_heatmap(X):
    return np.asarray([[weight_color_scale(x) for x in X_] for X_ in X])


def coef_interpolate(coef_table: dict, mirror=None):
    '''Returns value for coefficients interpolated between the keys in the given dictionary
    '''

    # add mirrored keys
    if mirror in [90]:
        coef_table.update({90 + (90 - k): v for k, v in coef_table.items()})
    if mirror in [90, 180]:
        coef_table.update({180 + (180 - k): v for k, v in coef_table.items()})

    # add replicated angles
    coef_table.update({360 + k: v for k, v in coef_table.items()})
    coef_table.update({0 - k: v for k, v in coef_table.items()})

    coef_keys = np.asarray(list(coef_table.keys()))

    def get_coef(x):
        diffs = coef_keys - x
        k1 = coef_keys[diffs <= 0][np.argmin(np.abs(diffs)[diffs <= 0])]
        k2 = coef_keys[diffs > 0][np.argmin(np.abs(diffs)[diffs > 0])]
        x1, x2 = np.abs(k1 - x), np.abs(k2 - x)
        v1, v2 = coef_table[k1], coef_table[k2]

        if x1 > 10:
            x1, v1 = 10, -0.01
        if x2 > 10:
            x2, v2 = 10, -0.01
        t = x1 / (x1 + x2 + 1e-3)
        return (1 - t) * v1 + t * v2

    return np.vectorize(get_coef)
