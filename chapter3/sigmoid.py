# coding: utf-8

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

if __name__ == '__main__':
    main()
