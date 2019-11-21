# A little introductory for you, folks.
# We are writing a program to solve multi-classification problem
# https://www.kaggle.com/c/cifar-10/overview dataset from here
# PLEASE, PAY ATTENTION!
# If you will create a folder for new lab, don't forget to include venv and .idea and another system
# directories to .gitignore

from datetime import datetime
from argparse import ArgumentParser
from keras.engine.saving import model_from_json
from keras.layers import BatchNormalization
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras import optimizers


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


def preprocessing_data(x_train, y_train, x_test, y_test):
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    x_train = x_train.reshape((50000, 32 * 32 * 3))
    x_test = x_test.reshape((10000, 32 * 32 * 3))
    x_train_cat = x_train.astype('float32') / 255
    x_test_cat = x_test.astype('float32') / 255
    return (x_train_cat, y_train_cat), (x_test_cat, y_test_cat)


def save_network(network):
    network.save_weights("model_weights.h5")
    model_json = network.to_json()
    with open("model_arch.json", "w") as json_file:
        json_file.write(model_json)


def load_and_run(path_to_model, path_to_weight):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    json_file = open(path_to_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_to_weight)
    loaded_model.compile(optimizer='sgd',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    score_train = loaded_model.evaluate(x_train, y_train, verbose=0)
    score_test = loaded_model.evaluate(x_test, y_test, verbose=0)
    return score_train, score_test


def fit_and_run(hidden_size=300, batch_size=128, rate=0.1, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    time_start = datetime.now()
    network = models.Sequential()
    network.add(BatchNormalization(input_shape=(32 * 32 * 3,)))
    network.add(layers.Dense(hidden_size, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                batch_size=batch_size,  verbose=2)
    save_network(network)
    delta_time = datetime.now() - time_start
    score_train = network.evaluate(x_train, y_train, verbose=0)
    score_test = network.evaluate(x_test, y_test, verbose=0)
    return delta_time, score_train, score_test


def print_info(time, score_train, score_test):
    print('Train loss: ', score_train[0])
    print('Train accuracy: ', score_train[1])
    print('Test loss: ', score_test[0])
    print('Test accuracy: ', score_test[1])
    print('Train time: ', time)


def main(args):
    if args.mode == 'fit':
        time, score_train, score_test = fit_and_run(args.hidden_num, args.batch_size, args.rate, args.epochs)
        print_info(time, score_train, score_test)
    else:
        score_train, score_test = load_and_run("model_arch.json", "model_weights.h5")
        print_info(0.0, score_train, score_test)


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())


