from keras import models, Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import SGD


def first_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(layers.Dense(128, activation='relu',  kernel_initializer='he_uniform'))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def second_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(layers.Dense(128, activation='relu',  kernel_initializer='he_uniform'))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def third_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def fourth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='tanh',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='tanh',
                       kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(layers.Dense(128, activation='tanh', kernel_initializer='he_normal'))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def fifth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='sigmoid',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='sigmoid',
                       kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(layers.Dense(128, activation='sigmoid', kernel_initializer='he_normal'))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def sixth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def seventh_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='tanh',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='tanh', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def eighth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='sigmoid',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def ninth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def tenth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.25))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def eleventh_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='tanh',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.25))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.1, decay=1e-2/20,  momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def twelfth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='sigmoid',
                       kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.25))
    network.add(Flatten())
    network.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.1, decay=1e-2/20,  momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def thirteenth_model():
    network = Sequential()
    network.add(Conv2D(32, (3, 3),
                       kernel_initializer='he_uniform', activation='relu', padding='same', input_shape=(32, 32, 3)))
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.3))
    network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.4))

    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.1, decay=1e-2 / 20, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def fourteenth_model():
    network = Sequential()
    network.add(BatchNormalization())
    network.add(Conv2D(32, (3, 3),
                       kernel_initializer='he_uniform',  padding='same', input_shape=(32, 32, 3)))
    network.add(LeakyReLU())
    network.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    network.add(LeakyReLU())
    network.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.3))
    network.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    network.add(LeakyReLU())
    network.add(Conv2D(128, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Dropout(0.4))

    network.add(Flatten())
    network.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=0.1, decay=1e-2 / 20, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return network


def get_all_models_with_names():
    return {'1': first_model, '2': second_model, '3': third_model, '4': fourth_model,
            '5': fifth_model, '6': sixth_model, '7': seventh_model, '8': eighth_model,
            '9': ninth_model, '10': tenth_model, '11': eleventh_model, '12': twelfth_model,
            '13': thirteenth_model, '14': fourteenth_model}


def choose_model(model_num='1'):
    return get_all_models_with_names()[model_num]




