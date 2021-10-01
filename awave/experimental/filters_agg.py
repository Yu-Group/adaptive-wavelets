'''Code here adapted from https://distill.pub/2020/circuits/curve-circuits/
'''

import numpy as np

from awave.experimental.filters import edge_filter, curve_filter
from awave.experimental.util import coef_interpolate


def edge_edge_connect(size, ang1, ang2):
    F_edge, F_surround = edge_filter(size, ang2)
    ang_diff = int(ang1 - ang2) % 180
    ang_diff = min(ang_diff, 180 - ang_diff)
    coef = {0: 1, 10: 0.2, 20: 0.01, 30: -0.0, 40: -0.05, 50: -0.05, 60: -0.1, 70: -0.1, 80: -0.3, 90: -0.6}[ang_diff]
    return coef * F_edge - 0.1 * np.maximum(0, coef) * F_surround


def curve_curve_connect(size, ang1, ang2, r):
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
    if isinstance(r, list):
        return np.mean([edge_curve_connect(size, ang1, ang2, r_) for r_ in r], axis=0)

    F, tangent_angles = curve_filter(size, ang2, r)
    ang_diff = tangent_angles - ang1

    coef1 = coef_interpolate(
        {0: 1, 10: 0.8, 20: 0.2, 30: -0.1, 40: -0.8, 50: -0.5, 60: -0.1},
        mirror=90)(ang_diff.astype("int32"))
    coef2 = coef_interpolate(
        {0: 1, 5: 1, 10: 0.5, 15: -1, 20: -1, 30: -1, 40: -0.8, 50: -0.4, 60: -0.1},
        mirror=90)(ang_diff.astype("int32"))

    coef = np.where(
        np.logical_xor(np.less_equal((ang1 - ang2) % 180, 90),
                       np.greater_equal((tangent_angles - ang2) % 180, 90)
                       ),

        coef1, coef2)

    return F * coef - 0.1 * (1 - F) * np.maximum(0, coef)



