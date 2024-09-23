if True:
    from reset_random import reset_random

    reset_random()
import math
import os
import shutil

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.callbacks import ModelCheckpoint

import ae
import config
import vae
from data_handler import get_train_test, load_data
from utils import TrainingCallback, print_df_to_table

matplotlib.use('Qt5Agg')
matplotlib.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'serif'

AE_PLOT = plt.figure(num=1)
MODELS = {
    'autoencoder': ae.build_autoencoder,
    'variational_autoencoder': vae.build_variationalAE,
}
DELETE_MODEL = False
BATCH_SIZE = {
    'autoencoder': 32,
    'variational_autoencoder': 64,
}

def evaluate_data(model, x, for_='Train'):
    print('[INFO] Evaluating {0} Data'.format(for_))
    train_decoded = model.predict(x)
    mse = mean_squared_error(x, train_decoded)
    mae = mean_absolute_error(x, train_decoded)
    df_ = pd.DataFrame(list(zip(['MSE', 'RMSE', 'MAE'], [mse, math.sqrt(mse), mae])), columns=['Metrics', 'Values'])
    df_['Values'] = df_['Values'].round(4)
    print_df_to_table(df_)
    return df_


def train(ae_name='variational_autoencoder'):
    df = load_data()
    train_x, test_x = get_train_test(df)

    model_dir = os.path.join('models', ae_name)
    if DELETE_MODEL:
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    encoder, decoder, autoencoder = MODELS[ae_name](train_x.shape[1])

    loss_csv_path = os.path.join(model_dir, 'loss.csv')
    model_path = os.path.join(model_dir, 'model.h5')
    training_cb = TrainingCallback(loss_csv_path, AE_PLOT)
    checkpoint = ModelCheckpoint(
        model_path, save_best_only=True, save_weights_only=True,
        monitor='val_loss', mode='min', verbose=False
    )

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(loss_csv_path):
        print('[INFO] Loading Pre-Trained Model :: {0}'.format(model_path))
        autoencoder.load_weights(model_path)
        initial_epoch = len(pd.read_csv(loss_csv_path))

    print('[INFO] Fitting Data')
    autoencoder.fit(
        train_x, train_x, validation_data=(test_x, test_x), epochs=30,
        batch_size=BATCH_SIZE[ae_name],
        verbose=0, initial_epoch=initial_epoch, callbacks=[training_cb, checkpoint]
    )

    print('[INFO] Encoding Data')
    encoded = encoder.predict(df.values[:, 1:])
    encoded_df = pd.DataFrame(encoded, columns=['Feature_{0}'.format(i + 1) for i in range(ae.ENCODER_DIM)])
    encoded_df.insert(0, 'Patients', df.values[:, 0])

    os.makedirs(config.AE_DATA_DIR, exist_ok=True)
    print('[INFO] Saving Encoded Data :: {0}'.format(config.AE_ENCODED_PATH))
    encoded_df.to_csv(config.AE_ENCODED_PATH, index=False)

    os.makedirs(config.AE_RESULTS_DIR, exist_ok=True)
    train_mdf = evaluate_data(autoencoder, train_x)
    train_mdf.to_csv(os.path.join(config.AE_RESULTS_DIR, 'Train.csv'), index=False)
    test_mdf = evaluate_data(autoencoder, test_x, for_='Test')
    test_mdf.to_csv(os.path.join(config.AE_RESULTS_DIR, 'Test.csv'), index=False)


if __name__ == '__main__':
    # train(ae_name='autoencoder')
    train(ae_name='variational_autoencoder')

