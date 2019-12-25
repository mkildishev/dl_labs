import datetime
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def preprocessing_data(x_train, y_train, x_test, y_test):
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    x_train_cat = x_train.astype('float32') / 255
    x_test_cat = x_test.astype('float32') / 255
    return (x_train_cat, y_train_cat), (x_test_cat, y_test_cat)


def save_network(network, name):
    weight_name = build_name_for_obj('.h5', '/models/', name)
    arch_name = build_name_for_obj('.json', '/models/', name)
    network.save_weights(weight_name)
    model_json = network.to_json()
    with open(arch_name, "w") as json_file:
        json_file.write(model_json)


def build_name_for_obj(format, path, *args):
    path = os.getcwd() + path
    if not os.path.exists(path):
        os.mkdir(path)
    name = path + '_'.join([*args])
    name += format
    return name


def save_comparision_plot(train_values, validation_values, y_label, model_name, plot_name, x_step, y_step):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
    plt.plot(train_values, marker='o')
    plt.plot(validation_values, marker='o')
    plt.title(plot_name)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig(build_name_for_obj('.png', '/plots/', model_name, plot_name))



def print_info(time, score_train, score_test):
    print('Train loss: ', score_train[0])
    print('Train accuracy: ', score_train[1])
    print('Test loss: ', score_test[0])
    print('Test accuracy: ', score_test[1])
    print('Train time: ', time)