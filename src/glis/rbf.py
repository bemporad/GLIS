"""
GLIS - (GL)obal optimization solvers using (I)nverse distance weighting and
radial basis function (S)urrogates.

RBF functions.

(C) 2019-2023 Alberto Bemporad, Mengjia Zhu

"""

import numpy as np
from math import exp

# RBFs (@njit's improvements do not seem significant here)
def inverse_quadratic(x1, x2, epsil):
    return 1. / (1. + epsil ** 2 * np.sum((x1 - x2) ** 2, axis=-1))


def gaussian(x1, x2, epsil):
    return np.exp(-epsil ** 2 * np.sum((x1 - x2) ** 2, axis=-1))


def multiquadric(x1, x2, epsil):
    return np.sqrt(1. + epsil ** 2 * np.sum((x1 - x2) ** 2, axis=-1))


def thin_plate_spline(x1, x2, epsil):
    return epsil ** 2 * np.sum((x1 - x2) ** 2, axis=-1) * np.log(epsil * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)))


def linear(x1, x2, epsil):
    return epsil * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))


def inverse_multi_quadric(x1, x2, epsil):
    return 1. / np.sqrt(1. + epsil ** 2 * np.sum((x1 - x2) ** 2, axis=-1))

