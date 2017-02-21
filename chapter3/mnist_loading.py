# coding: utf-8

import sys
import os
import pickle

from PIL import Image
import numpy as np

from sigmoid import sigmoid
from softmax import softmax

sys.path.append(os.pardir)
from mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(a2, W3) + b3
    return softmax(a3)


def img_show(img):
    pimg = Image.fromarray(np.uint8(img))
    pimg.show()


def main():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        expect = t[1]
        if p == expect:
            accuracy_cnt += 1
        else:
            #img_show(x[i].reshape(28, 28))
            print("Predict: {}".format(p))
            print("Expect: {}".format(expect))
    print("Accuracy: {}".format(float(accuracy_cnt) / len(x)))
if __name__ == '__main__':
    main()
