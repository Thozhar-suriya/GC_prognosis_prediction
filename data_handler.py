import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_data():
    print('[INFO] Loading Data From :: {0}'.format(config.DATA_PATH))
    df = pd.read_csv(config.DATA_PATH, low_memory=False)
    return df


def get_train_test(df):
    print('[INFO] Splitting Data Into Train|Test')
    train_x, test_x = train_test_split(df.values[:, 1:], test_size=0.3, shuffle=False, random_state=1)
    print('[INFO] Train Shape :: {0}'.format(train_x.shape))
    print('[INFO] Test Shape :: {0}'.format(test_x.shape))
    return train_x, test_x


if __name__ == '__main__':
    get_train_test(load_data())
