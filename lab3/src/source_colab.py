
from models import *
from utils import *
from datetime import datetime
from argparse import ArgumentParser
from keras.utils import plot_model
from keras.engine.saving import model_from_json
from keras.datasets import cifar10



def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['load', 'fit', 'fit_all'], default='fit_all',
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


def fit_all(models, batch_size=128, rate=0.1, epochs=20):
    for model_name, model in models.items():
        print('Model ' + model_name)
        fit_and_run(model, model_name, batch_size, rate, epochs)


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


def fit_and_run(model=first_model, name="model", batch_size=128, rate=0.1, epochs=20):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = preprocessing_data(x_train, y_train, x_test, y_test)
    network = model()
    time_start = datetime.now()
    history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                          batch_size=batch_size,  verbose=2)
    delta_time = datetime.now() - time_start
    save_comparision_plot(history.history['acc'], history.history['val_acc'], 'Accuracy',
                          name, 'Network accuracy')
    save_comparision_plot(history.history['loss'], history.history['val_loss'], 'Loss',
                          name, 'Network loss')
    save_network(network, name)
    plot_model(network, build_name_for_obj('.png', '/models/', name), show_shapes=True, dpi=200)
    score_train = network.evaluate(x_train, y_train, verbose=0)
    score_test = network.evaluate(x_test, y_test, verbose=0)
    print_info(delta_time, score_train, score_test)


def main(args):
    if args.mode == 'fit':
        fit_and_run(choose_model(args.model), args.model, args.batch_size, args.rate, args.epochs)
    elif args.mode == 'fit_all':
        fit_all(get_all_models_with_names(), args.batch_size, args.rate, args.epochs)
    else:
        model_arch = build_name_for_obj('.json', '/models/', args.model)
        model_weights = build_name_for_obj('.h5', '/models/', args.model)
        load_and_run(model_arch, model_weights)


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
