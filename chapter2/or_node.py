# coding: utf-8

import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return int(np.sum(w * x) + b > 0)


def main():
    assert OR(0, 0) == 0
    assert OR(1, 0) == 1
    assert OR(0, 1) == 1
    assert OR(1, 1) == 1

if __name__ == '__main__':
    main()
