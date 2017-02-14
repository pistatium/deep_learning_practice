# coding: utf-8

import sys
import os

from PIL import Image

sys.path.append(os.pardir)
from mnist import load_mnist

def img_show(img):
    pass

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    print(x_train)
    print(t_train)
    print(x_test)
    print(t_test)


if __name__ == '__main__':
    main()
