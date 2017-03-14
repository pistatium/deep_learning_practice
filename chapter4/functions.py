# coding: utf-8

import numpy as np


def x_entropy_error(y: np.core.multiarray, t: np.core.multiarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

    return -np.sum(t * np.log(y)) /batch_size
