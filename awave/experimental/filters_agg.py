'''Code here adapted from https://distill.pub/2020/circuits/curve-circuits/
'''

import numpy as np

from awave.experimental.filters import edge_filter, curve_filter, gabor_filter
from awave.experimental.util import coef_interpolate


def edge_edge_connect(size, ang1, ang2):
    '''Connect 2 edge filters
    '''
    F_edge, F_surround = edge_filter(size, ang2)
    ang_diff = int(ang1 - ang2) % 180
    ang_diff = min(ang_diff, 180 - ang_diff)
    coef = {0: 1, 10: 0.2, 20: 0.01, 30: -0.0, 40: -0.05, 50: -0.05, 60: -0.1, 70: -0.1, 80: -0.3, 90: -0.6}[ang_diff]
    return coef * F_edge - 0.1 * np.maximum(0, coef) * F_surround


def curve_curve_connect(size, ang1, ang2, r):
    '''Connect 2 curve filters
    '''
    if isinstance(r, list):
        return np.mean([curve_curve_connect(size, ang1, ang2, r_) for r_ in r], axis=0)

    F, tangent_angles = curve_filter(size, ang2, r)
    ang_diff = tangent_angles - ang1

    get_coef = coef_interpolate({
        0: 1, 10: 0.8, 20: 0.7, 30: 0.4, 40: 0.2,
        130: -0.05, 140: -0.2, 150: -0.4, 160: -0.8, 170: -1, 180: -1},
        mirror=180)

    coef = 0.8 * get_coef(tangent_angles - ang1) + 0.2 * get_coef(ang2 - ang1)
    return F * coef - 0.1 * (1 - F) * np.maximum(0, coef)


def edge_curve_connect(size, ang1, ang2, r):
    '''Connect an edge and a curve filter
    '''
    if isinstance(r, list):
        return np.mean([edge_curve_connect(size, ang1, ang2, r_) for r_ in r], axis=0)

    F, tangent_angles = curve_filter(size, ang2, r)
    ang_diff = tangent_angles - ang1

    coef1 = coef_interpolate(
        {0: 1, 10: 0.8, 20: 0.2, 30: -0.1, 40: -0.8, 50: -0.5, 60: -0.1
         }, mirror=90)(ang_diff.astype("int32"))
    coef2 = coef_interpolate(
        {0: 1, 5: 1, 10: 0.5, 15: -1, 20: -1, 30: -1, 40: -0.8, 50: -0.4, 60: -0.1
         }, mirror=90)(ang_diff.astype("int32"))

    coef = np.where(
        np.logical_xor(np.less_equal((ang1 - ang2) % 180, 90),
                       np.greater_equal((tangent_angles - ang2) % 180, 90)
                       ),

        coef1, coef2)

    return F * coef - 0.1 * (1 - F) * np.maximum(0, coef)

def make_weight_connection(size, in_spec, out_spec, r=None):
    if in_spec[0] == "color" and out_spec[0] == "gabor":
        F = gabor_filter(size, angle=out_spec[1], shift=out_spec[2])

    elif in_spec[0] in ("gabor", "edge") and out_spec[0] == "edge":
        F = edge_edge_connect(size, ang1=in_spec[1], ang2=out_spec[1])

    elif in_spec[0] == "edge" and out_spec[0] == "curve":
        assert r != None
        F = edge_curve_connect(size, ang1=in_spec[1], ang2=out_spec[1], r=r)

    elif in_spec[0] == "curve" and out_spec[0] == "curve":
        assert r != None
        F = curve_curve_connect(size, ang1=in_spec[1], ang2=out_spec[1], r=r)

    else:
        if (in_spec[0], out_spec[0]) not in _warned:
            print("Warning: no registered map", in_spec[0], '->', out_spec[0])
            # _warned.append((in_spec[0], out_spec[0]))
        F = np.zeros([size, size])
    return F[..., None, None]


def make_weights(size, in_specs, out_specs, r=None):
    '''Note: these weights output are W X H x input_channels x output_channels (would usually transpose this)
    '''
    W = np.concatenate([
        np.concatenate([make_weight_connection(size, in_spec, out_spec, r=r) for in_spec in in_specs], axis=-2)
        for out_spec in out_specs], axis=-1)
    # W /= np.sqrt((W**2).sum(axis=(0,1,2), keepdims=True) + 1e-3)
    return W