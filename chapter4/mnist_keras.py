# coding: utf-8

import sys
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape

sys.path.append(os.pardir)
from mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True, normalize=False)
    model = Sequential()

    model.add(Reshape((28, 28 ,1), input_shape=(1, 28, 28)))
    model.add(Conv2D(filters=12, padding='same', kernel_size=(3, 3)))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(output_dim=20))
    model.add(Dropout(0.1))
    model.add(Activation("sigmoid"))

    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, t_train, nb_epoch=30, batch_size=100)
    loss, accuracy = model.evaluate(x_test, t_test, verbose=0)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    print(model.predict(x_test[0:3], batch_size=1, verbose=0))

if __name__ == '__main__':
    main()
