# coding: utf-8

import numpy as np

from chapter3 import softmax
from functions import x_entropy_error


class TwoLayerNet(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, std: float=0.01):
        self.params = {
            'W1': std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(hidden_size),
        }

    def predict(self, x: np.core.multiarray):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = softmax(a1)
        a2 = np.dot(x, W2) + b2
        return softmax(a2)

    def loss(self, x: np.core.multiarray, t: np.core.multiarray):
        y = self.predict(x)
        return x_entropy_error(y, t)

    def accuracy(self, x: np.core.multiarray, t: np.core.multiarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

