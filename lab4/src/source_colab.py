
from models import *
from utils import *
from datetime import datetime
from argparse import ArgumentParser
from keras.utils import plot_model
from keras.engine.saving import model_from_json
from keras.datasets import cifar10

def fit_and_run_ae_first(batch_size=128, rate=0.1, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    encoder, decoder, autoencoder = first_model_ae()
    optimizer = optimizers.SGD(lr=rate, decay=1e-2 / 10, momentum=0.9)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    time_start = datetime.now()
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    delta_time = datetime.now() - time_start
    score_train = autoencoder.evaluate(x_train, x_train, verbose=0)
    score_test = autoencoder.evaluate(x_test, x_test, verbose=0)
    print_info(delta_time, score_train, score_test)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'First ae accuracy', 1, 0.005)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'First ae loss', 1, 0.005)
    model_pretrained = Sequential()
    for layer in encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
    model_pretrained.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = datetime.now()
    history = model_pretrained.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'First FCNN accuracy', 1, 0.05)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'First FCNN loss', 1, 0.05)
    save_network(model_pretrained, "first_model_with_ae")
    delta_time = datetime.now() - time_start
    score_train = model_pretrained.evaluate(x_train, y_train, verbose=0)
    score_test = model_pretrained.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)


def fit_and_run_ae_second(batch_size=128, rate=0.001, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    encoder, decoder, autoencoder = second_model_ae()
    optimizer = optimizers.SGD(lr=rate, decay=1e-2 / 10, momentum=0.9)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    time_start = datetime.now()
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    delta_time = datetime.now() - time_start
    score_train = autoencoder.evaluate(x_train, x_train, verbose=0)
    score_test = autoencoder.evaluate(x_test, x_test, verbose=0)
    print_info(delta_time, score_train, score_test)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Second ae accuracy', 1, 0.005)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Second ae loss', 1, 0.005)
    model_pretrained = Sequential()
    for layer in encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Flatten())
    model_pretrained.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model_pretrained.add(Dense(10, activation='softmax', kernel_initializer='he_uniform'))
    model_pretrained.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = datetime.now()
    history = model_pretrained.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Second FCNN accuracy', 1, 0.05)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Second FCNN loss', 1, 0.05)
    save_network(model_pretrained, "second_model_with_ae")
    delta_time = datetime.now() - time_start
    score_train = model_pretrained.evaluate(x_train, y_train, verbose=0)
    score_test = model_pretrained.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)


def fit_and_run_ae_third(batch_size=128, rate=0.001, epochs=2):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    encoder, decoder, autoencoder = third_model_ae()
    optimizer = optimizers.SGD(lr=rate, decay=1e-2 / 10, momentum=0.9)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    time_start = datetime.now()
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    delta_time = datetime.now() - time_start
    score_train = autoencoder.evaluate(x_train, x_train, verbose=0)
    score_test = autoencoder.evaluate(x_test, x_test, verbose=0)
    print_info(delta_time, score_train, score_test)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Third ae accuracy', 1, 0.005)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Third ae loss', 1, 0.005)
    model_pretrained = Sequential()
    for layer in encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Flatten())
    model_pretrained.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model_pretrained.add(Dense(10, activation='softmax', kernel_initializer='he_uniform'))
    model_pretrained.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = datetime.now()
    history = model_pretrained.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Third FCNN accuracy', 1, 0.05)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Third FCNN loss', 1, 0.05)
    save_network(model_pretrained, "second_model_with_ae")
    delta_time = datetime.now() - time_start
    score_train = model_pretrained.evaluate(x_train, y_train, verbose=0)
    score_test = model_pretrained.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)




if __name__ == '__main__':

