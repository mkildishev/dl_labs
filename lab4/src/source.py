# A little introductory for you, folks.
# We are writing a program to solve multi-classification problem
# https://www.kaggle.com/c/cifar-10/overview dataset from here
# PLEASE, PAY ATTENTION!
# If you will create a folder for new lab, don't forget to include venv and .idea and another system
# directories to .gitignore
from keras import optimizers

from src.models import *
from src.utils import *
from datetime import datetime
from argparse import ArgumentParser
from keras.engine.saving import model_from_json
from keras.datasets import cifar10


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['load', 'fit', 'fit_all'], default='fit',
                        help='Choosing work mode')
    parser.add_argument('--model', type=str,  default='11',
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Sample batch size')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs count')
    return parser


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


def fit_and_run_ae_third(batch_size=128, rate=0.001, epochs=20):
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


def fit_and_run_ae_fourth(batch_size=128, rate=0.001, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    optimizer = optimizers.SGD(lr=rate, decay=1e-2 / 10, momentum=0.9)
    encoders, autoencoders = fourth_model_ae()
    for autoencoder in autoencoders:
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    time_start = datetime.now()
    history = autoencoders[0].fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    delta_time = datetime.now() - time_start
    new_x_train = encoders[0].predict(x_train)
    new_x_test = encoders[0].predict(x_test)
    history = autoencoders[1].fit(new_x_train, new_x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                  validation_data=(new_x_test, new_x_test), verbose=2)
    new_x_train = encoders[1].predict(new_x_train)
    new_x_test = encoders[1].predict(new_x_test)
    history = autoencoders[2].fit(new_x_train, new_x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                  validation_data=(new_x_test, new_x_test), verbose=2)

    model_pretrained = Sequential()
    for layer in encoders:
        model_pretrained.add(layer)
    model_pretrained.add(Dense(10, activation='softmax', kernel_initializer='he_uniform'))
    model_pretrained.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = datetime.now()
    history = model_pretrained.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Fourth FCNN accuracy', 1, 0.05)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Fourth FCNN loss', 1, 0.05)
    save_network(model_pretrained, "fourth_model_with_ae")
    delta_time = datetime.now() - time_start
    score_train = model_pretrained.evaluate(x_train, y_train, verbose=0)
    score_test = model_pretrained.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)

def fit_and_run_ae_fifth(batch_size=80, rate=0.1, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    noiser, encoder, decoder, denoiser_model = fifth_model_ae()
    optimizer = optimizers.SGD(lr=rate, decay=1e-2 / 10, momentum=0.9)
    denoiser_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    time_start = datetime.now()
    history = denoiser_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    delta_time = datetime.now() - time_start
    noised_x_train = noiser.predict(x_train, 80)
    noised_x_test = noiser.predict(x_test, 80)
    score_train = denoiser_model.evaluate(noised_x_train, x_train, batch_size=80, verbose=0)
    score_test = denoiser_model.evaluate(noised_x_test, x_test, batch_size=80, verbose=0)
    print_info(delta_time, score_train, score_test)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Fifth ae accuracy', 1, 0.005)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Fifth ae loss', 1, 0.005)
    model_pretrained = Sequential()
    for layer in encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
    model_pretrained.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = datetime.now()
    history = model_pretrained.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    # Save plots & model
    save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
                          "", 'Fifth FCNN accuracy', 1, 0.05)
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          "", 'Fifth FCNN loss', 1, 0.05)
    save_network(model_pretrained, "first_model_with_ae")
    delta_time = datetime.now() - time_start
    score_train = model_pretrained.evaluate(x_train, y_train, verbose=0)
    score_test = model_pretrained.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)


# def fit_all(models, batch_size=128, rate=0.1, epochs=20):
#     for model_name, model in models.items():
#         print('Model ' + model_name)
#         fit_and_run(model, model_name, batch_size, rate, epochs)


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
    print_info(0.0, score_train, score_test)


# def fit_and_run(model=second_model, name="model", batch_size=128, rate=0.1, epochs=20):
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
#     network = model()
#     time_start = datetime.now()
#     history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
#                           batch_size=batch_size,  verbose=2)
#     delta_time = datetime.now() - time_start
#     save_comparision_plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy',
#                           name, 'Network accuracy')
#     save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
#                           name, 'Network loss')
#     save_network(network, name)
#     score_train = network.evaluate(x_train, y_train, verbose=0)
#     score_test = network.evaluate(x_test, y_test, verbose=0)
#     print_info(delta_time, score_train, score_test)


# def main(args):
#     if args.mode == 'fit':
#         fit_and_run(choose_model(args.model), args.model, args.batch_size, args.rate, args.epochs)
#     elif args.mode == 'fit_all':
#         fit_all(get_all_models_with_names(), args.batch_size, args.rate, args.epochs)
#     else:
#         model_arch = build_name_for_obj('.json', '/models/', args.model)
#         model_weights = build_name_for_obj('.h5', '/models/', args.model)
#         load_and_run(model_arch, model_weights)


if __name__ == '__main__':
    # parser = create_parser()
    # main(parser.parse_args())
    fit_and_run_ae_fourth()