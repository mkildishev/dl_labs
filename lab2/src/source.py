# A little introductory for you, folks.
# We are writing a program to solve multi-classification problem
# https://www.kaggle.com/c/cifar-10/overview dataset from here
# PLEASE, PAY ATTENTION!
# If you will create a folder for new lab, don't forget to include venv and .idea and another system
# directories to .gitignore

from argparse import ArgumentParser
import numpy as np
from numpy import load
from keras import models
from keras.datasets import mnist
from keras import layers
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras import optimizers


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--hidden_num', type=int,  default=50000,
                        help='Hidden layer nodes num')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Sample batch size')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs count')
    return parser


def preprocessing_data(x_train, y_train, x_test, y_test):
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    x_train = x_train.reshape((50000, 32 * 32 * 3))
    x_test = x_test.reshape((10000, 32 * 32 * 3))
    x_train_cat = x_train.astype('float32') / 255
    x_test_cat = x_test.astype('float32') / 255
    return (x_train_cat, y_train_cat), (x_test_cat, y_test_cat)


def run(hidden_size=1284, batch_size=128, rate=0.1, epochs=15):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    network = models.Sequential()
    network.add(layers.Dense(hidden_size, activation='relu',kernel_initializer='he_normal', input_shape=(32 * 32 * 3,)))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = optimizers.SGD(rate)
    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    network.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
    train_result = network.evaluate(x_train, y_train)
    test_result = network.evaluate(x_test, y_test)

    return train_result, test_result


def main(args):
    pass


if __name__ == '__main__':
    # parser = create_parser()
    # main(parser.parse_args())
    print(run())


