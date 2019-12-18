from keras.layers import BatchNormalization
from keras import models
from keras import layers
from keras import optimizers


def first_model(rate, epochs):
    network = models.Sequential()
    network.add(layers.Dense(300, activation='relu', input_shape=(32 * 32 * 3,), kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def second_model(rate, epochs):
    network = models.Sequential()
    network.add(layers.Dense(300, activation='relu', input_shape=(32 * 32 * 3,), kernel_initializer='he_normal'))
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
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def fourth_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='tanh', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def fifth_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='sigmoid', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def sixth_model(rate, epochs):
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


def seventh_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='tanh', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def eighth_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='sigmoid', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def ninth_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='tanh', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def tenth_model(rate, epochs):
    network = models.Sequential()
    network.add(BatchNormalization())
    network.add(layers.Dense(300, activation='sigmoid', kernel_initializer='he_normal'))
    network.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10,  activation='softmax'))
    optimizer = optimizers.SGD(lr=rate, decay=1e-2/epochs)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def get_all_models_with_names():
    return {'1': first_model, '2': second_model, '3': third_model, '4': fourth_model,
            '5': fifth_model, '6': sixth_model, '7': seventh_model, '8': eighth_model,
            '9': ninth_model, '10': tenth_model}


def choose_model(model_num='1'):
    return get_all_models_with_names()[model_num]




