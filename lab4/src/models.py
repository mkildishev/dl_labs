from keras import models, Sequential, Input, Model
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, Reshape, \
    UpSampling2D
from keras.optimizers import SGD
import keras.backend as K
from keras.layers import Lambda


def first_model_ae(): # Common autoencoder
    input_img = Input(shape=(32, 32, 3))
    flat_img = Flatten()(input_img)
    layer = Dense(300, kernel_initializer='he_normal', activation='relu')(flat_img)
    encoded = Dense(128, kernel_initializer='he_normal', activation='relu')(layer)

    input_encoded = Input(shape=(128,))
    layer = Dense(300, kernel_initializer='he_normal', activation='relu')(input_encoded)
    flat_decoded = Dense(32 * 32 * 3, kernel_initializer='he_normal', activation='relu')(layer)
    decoded = Reshape((32, 32, 3))(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def second_model_ae(): # Conv. autoencoder
    input_img = Input(shape=(32, 32, 3))
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(
        input_img)
    encoded = MaxPooling2D((2, 2))(layer)

    input_encoded = Input(shape=(16, 16, 32))
    layer = UpSampling2D((2, 2))(input_encoded)
    decoded = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')
    return encoder, decoder, autoencoder


def third_model_ae(): # Conv. deep autoencoder
    input_img = Input(shape=(32, 32, 3))
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(
        input_img)
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    layer = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    encoded = MaxPooling2D(pool_size=2)(layer)

    input_encoded = Input(shape=(8, 8, 64))
    layer = UpSampling2D((2, 2))(input_encoded)
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    layer = UpSampling2D((2, 2))(layer)
    layer = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)
    decoded = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(layer)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')
    return encoder, decoder, autoencoder


def fourth_model_ae(): # Stack autoencoder
    input_img = Input(shape=(32, 32, 3))
    flat_img = Flatten()(input_img)
    encoded_1 = Dense(300, activation='relu', kernel_initializer='he_uniform')(flat_img)

    input_encoded_1 = Input(shape=(300,))
    flat_decoded = Dense(32 * 32 * 3, activation='relu', kernel_initializer='he_uniform')(input_encoded_1)
    decoded_1 = Reshape((32, 32, 3))(flat_decoded)

    encoder_1 = Model(input_img, encoded_1, name='encoder_1_stack')
    decoder_1 = Model(input_encoded_1, decoded_1, name='decoder_1_stack')
    autoencoder_1 = Model(input_img, decoder_1(encoder_1(input_img)), name='autoencoder_1_stack')


    input_2 = Input(shape=(300,))
    encoded_2 = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_2)

    input_encoded_2 = Input(shape=(128,))
    decoded_2 = Dense(300, activation='relu', kernel_initializer='he_uniform')(input_encoded_2)

    encoder_2 = Model(input_2, encoded_2, name='encoder_2_stack')
    decoder_2 = Model(input_encoded_2, decoded_2, name='decoder_2_stack')
    autoencoder_2 = Model(input_2, decoder_2(encoder_2(input_2)), name='autoencoder_2_stack')


    input_3 = Input(shape=(128,))
    encoded_3 = Dense(64, activation='relu', kernel_initializer='he_uniform')(input_3)

    input_encoded_3 = Input(shape=(64,))
    decoded_3 = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_encoded_3)

    encoder_3 = Model(input_3, encoded_3, name='encoder_3_stack')
    decoder_3 = Model(input_encoded_3, decoded_3, name='decoder_3_stack')
    autoencoder_3 = Model(input_3, decoder_3(encoder_3(input_3)), name='autoencoder_3_stack')

    return [encoder_1, encoder_2, encoder_3], [autoencoder_1, autoencoder_2, autoencoder_3]


def fifth_model_ae(): # Denoising autoencoder
    encoder, decoder, autoencoder = first_model_ae()
    def add_noise(x):
        noise_factor = 0.5
        x = x + K.random_normal(x.get_shape(), 0.5, noise_factor)
        x = K.clip(x, 0., 1.)
        return x

    input_img  = Input(batch_shape=(80, 32, 32, 3))
    noised_img = Lambda(add_noise)(input_img)

    noiser = Model(input_img, noised_img, name="noiser")
    denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
    return noiser, encoder, decoder, denoiser_model










def get_all_models_with_names():
    return {}# {'1': first_model_ae, '2': second_model, '3': third_model, '4': fourth_model,
            #'5': fifth_model, '6': sixth_model, '7': seventh_model, '8': eighth_model,
           # '9': ninth_model, '10': tenth_model, '11': eleventh_model, '12': twelfth_model,
          #  '13': thirteenth_model, '14': fourteenth_model, '15': fifteenth_model, '16': sixteenth_model}


def choose_model(model_num='1'):
    return get_all_models_with_names()[model_num]




