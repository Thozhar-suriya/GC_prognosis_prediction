if True:
    from reset_random import reset_random

    reset_random()
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Input, Layer, Lambda, Multiply, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

from utils import save_model_architecture

ENCODER_DIM = 100
EPS_STD = 1.0


def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


def build_encoder(input_dim):
    x = Input(shape=(input_dim,), name='x')
    h = Dense(ENCODER_DIM, activation='relu', name='hidden_layer')(x)

    z_mu = Dense(ENCODER_DIM, name='mu')(h)
    z_log_var = Dense(ENCODER_DIM, name='log_var')(h)

    z_mu, z_log_var = KLDivergenceLayer(name='kl')([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5 * t), name='sigma')(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=EPS_STD, shape=(K.shape(x)[0], ENCODER_DIM)), name='eps')
    z_eps = Multiply(name='z_eps')([z_sigma, eps])
    z = Add(name='z')([z_mu, z_eps])
    encoder = Model(x, z_mu)
    return encoder, x, eps, z


def build_decoder(input_dim):
    print('[INFO] Building Decoder Model')
    model = Sequential(name='decoder')
    model.add(Dense(input_dim, activation='sigmoid', input_dim=ENCODER_DIM))
    return model


def build_variationalAE(input_dim):
    encoder, x, eps, z = build_encoder(input_dim)
    decoder = build_decoder(input_dim)

    print('[INFO] Building Variational AutoEncoder')
    x_pred = decoder(z)
    model = Model(inputs=[x, eps], outputs=x_pred, name='variational_autoencoder')
    model.compile(optimizer=RMSProp(learning_rate=0.001), loss=nll)
    save_model_architecture(model, 'models/variational_autoencoder/model')
    return encoder, decoder, model


if __name__ == '__main__':
    build_variationalAE(1000)
