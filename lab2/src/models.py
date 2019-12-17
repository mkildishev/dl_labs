from keras.layers import BatchNormalization
from keras import models
from keras import layers
from keras import optimizers


def first_model(rate, epochs):
    network = models.Sequential()
    network.add(layers.Dense(300, activation='relu', input_shape=(32 * 32 * 3,), kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def second_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def third_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def choose_model(model_num='1'):
    if model_num == '1':
        return first_model
    elif model_num == '2':
        return second_model
    else:
        return third_model


def get_all_models_with_names():
    return {'1': first_model, '2': second_model, '3': third_model}

