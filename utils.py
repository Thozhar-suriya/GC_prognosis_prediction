import os
from contextlib import redirect_stdout

import matplotlib
import pandas as pd
import prettytable
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils.vis_utils import plot_model

matplotlib.use('Qt5Agg')
matplotlib.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'JetBrains Mono'


def plot_line(plt_, y1, y2, epochs, for_, save_path):
    ax = plt_.gca()
    ax.clear()
    ax.plot(range(epochs), y1, label='Training', color='dodgerblue')
    ax.plot(range(epochs), y2, label='Validation', color='orange')
    ax.set_title('Training and Validation {0}'.format(for_))
    ax.set_xlabel('Epochs')
    ax.set_ylabel(for_)
    ax.set_xlim([0, epochs])
    ax.legend()
    plt_.tight_layout()
    plt_.savefig(save_path)


class TrainingCallback(Callback):
    def __init__(self, loss_path, plt_):
        self.loss_path = loss_path
        self.plt_ = plt_
        if os.path.isfile(self.loss_path):
            self.df = pd.read_csv(self.loss_path)
            self.plot()
        else:
            self.df = pd.DataFrame([], columns=['epoch', 'loss', 'val_loss'])
            self.df.to_csv(self.loss_path, index=False)
        Callback.__init__(self)

    def plot(self):
        y1 = self.df['loss'].values.ravel()
        y2 = self.df['val_loss'].values.ravel()
        plot_line(self.plt_, y1, y2, len(self.df), 'Loss', self.loss_path.replace('.csv', '.png'))

    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[len(self.df.index)] = [
            int(epoch + 1), round(logs['loss'], 4), round(logs['val_loss'], 4),
        ]
        self.df.to_csv(self.loss_path, index=False)
        print('[Epoch:: {0}] -> Loss :: {1} | Val_Loss :: {2}'.format(
            epoch + 1, *[format(v, '.4f') for v in self.df.values[-1][1:]]
        ))
        self.plot()


def print_df_to_table(df, p=True):
    field_names = list(df.columns)
    p_table = prettytable.PrettyTable(field_names=field_names)
    p_table.add_rows(df.values.tolist())
    d = '\n'.join(['\t\t{0}'.format(p_) for p_ in p_table.get_string().splitlines(keepends=False)])
    if p:
        print(d)
    return d


def save_model_architecture(model, path):
    with open('{0}.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()
    plot_model(model, '{0}.png'.format(path), show_shapes=True)
