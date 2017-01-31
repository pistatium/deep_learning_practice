# coding: utf-8

import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return int(np.sum(w * x) + b > 0)


def main():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1

if __name__ == '__main__':
    main()
