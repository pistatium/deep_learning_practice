# coding: utf-8

import numpy as np


def x_entropy_error(y: np.core.multiarray, t: np.core.multiarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

    return -np.sum(t * np.log(y)) /batch_size


def numerical_gradient(f, x: np.core.multiarray) -> float:
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp
    return grad
