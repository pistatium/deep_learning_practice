# coding: utf-8

import numpy as np


def softmax(x):
    c = np.max(x)
    expx = np.exp(x - c)
    sum_expx = np.sum(expx)
    return expx / sum_expx
