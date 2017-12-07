# coding: utf-8

import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return int(np.sum(w * x) + b > 0)


def main():
    assert NAND(0, 0) == 1
    assert NAND(1, 0) == 1
    assert NAND(0, 1) == 1
    assert NAND(1, 1) == 0

if __name__ == '__main__':
    main()
