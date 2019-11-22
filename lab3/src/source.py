# A little introductory for you, folks.
# We are writing a program to solve multi-classification problem
# https://www.kaggle.com/c/cifar-10/overview dataset from here
# PLEASE, PAY ATTENTION!
# If you will create a folder for new lab, don't forget to include venv and .idea and another system
# directories to .gitignore

from argparse import ArgumentParser
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import cifar10


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['load', 'fit'], default='fit',
                        help='Choosing work mode')
    parser.add_argument('--hidden_num', type=int,  default=300,
                        help='Hidden layer nodes num')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Sample batch size')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs count')
    return parser


def save_network(network):
    network.save_weights("model_weights.h5")
    model_json = network.to_json()
    with open("model_arch.json", "w") as json_file:
        json_file.write(model_json)


def preprocessing_data(x_train, y_train, x_test, y_test):
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    x_train_cat = x_train.astype('float32') / 255
    x_test_cat = x_test.astype('float32') / 255
    return (x_train_cat, y_train_cat), (x_test_cat, y_test_cat)


def print_info(time, score_train, score_test):
    print('Train loss: ', score_train[0])
    print('Train accuracy: ', score_train[1])
    print('Test loss: ', score_test[0])
    print('Test accuracy: ', score_test[1])
    print('Train time: ', time)


def load_and_run(path_to_model, path_to_weight):
    pass


def fit_and_run(hidden_size=300, batch_size=128, rate=0.1, epochs=20):
    pass




if __name__ == '__main__':
    parser = create_parser()
