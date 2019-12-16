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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['load', 'fit'], default='fit',
                        help='Choosing work mode')
    parser.add_argument('--model', type=str,  default='2',
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Sample batch size')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs count')
    return parser


def first_model():
    network = models.Sequential()
    network.add(layers.Dense(300, activation='relu', input_shape=(32 * 32 * 3,), kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    return network


def second_model():
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    return network


def third_model():
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    return network


def preprocessing_data(x_train, y_train, x_test, y_test):
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    x_train = x_train.reshape((50000, 32 * 32 * 3))
    x_test = x_test.reshape((10000, 32 * 32 * 3))
    x_train_cat = x_train.astype('float32') / 255
    x_test_cat = x_test.astype('float32') / 255
    return (x_train_cat, y_train_cat), (x_test_cat, y_test_cat)


def save_network(network, name):
    weight_name = '.'.join([name, "h5"])
    arch_name = '.'.join([name, "json"])
    network.save_weights(weight_name)
    model_json = network.to_json()
    with open(arch_name, "w") as json_file:
        json_file.write(model_json)


def build_name_for_obj(format, *args):
    name = '_'.join([*args])
    name += format
    return name


def save_comparision_plot(train_values, validation_values, y_label, model_name, plot_name):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.plot(train_values, marker='o')
    plt.plot(validation_values, marker='o')
    plt.title(plot_name)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig(build_name_for_obj('.png', model_name, plot_name))


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


def fit_and_run(model=first_model, name="model", batch_size=128, rate=0.1, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    network = model()
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    time_start = datetime.now()
    history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                batch_size=batch_size,  verbose=2)

    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          name, 'Network accuracy')
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          name, 'Network loss')
    save_network(network, name)
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


def choose_model(model_num='1'):
    if model_num == '1':
        return first_model
    elif model_num == '2':
        return second_model
    else:
        return third_model


def main(args):
    if args.mode == 'fit':
        time, score_train, score_test = fit_and_run(choose_model(args.model), args.model, args.batch_size, args.rate, args.epochs)
        print_info(time, score_train, score_test)
    else:
        dot = '.'
        model_name = str(args.model)
        model_arch = dot.join([model_name, "json"])
        model_weights = dot.join([model_name, "h5"])
        score_train, score_test = load_and_run(model_arch, model_weights)
        print_info(0.0, score_train, score_test)


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())


