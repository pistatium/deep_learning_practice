# coding: utf-8

import sys
import os

from keras.models import Sequential
from keras.layers import Dense, Activation

sys.path.append(os.pardir)
from mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True, normalize=False)
    model = Sequential()
    model.add(Dense(output_dim=50, input_dim=784))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=100))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, t_train, nb_epoch=10, batch_size=100)
    loss, accuracy = model.evaluate(x_test, t_test, verbose=0)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    print(model.predict(x_test[0:3], batch_size=1, verbose=0))

if __name__ == '__main__':
    main()
