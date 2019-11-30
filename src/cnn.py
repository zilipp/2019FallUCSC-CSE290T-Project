from os import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from preprocess import preprocess
import logging
from utils import data_dir, init_logger


class CNN:
    _model = None
    _feature_size = 0

    def __init__(self, feature_size):
        self._feature_size = feature_size

        logging.info(tf.config.experimental.list_physical_devices('GPU'))

        inputs = keras.Input(shape=(feature_size, ))
        x = layers.Reshape((feature_size, 1),
                           input_shape=(feature_size, ))(inputs)
        x = layers.Conv1D(50, kernel_size=10, activation='relu')(x)
        x = layers.MaxPool1D(5)(x)
        x = layers.Conv1D(100, kernel_size=5, activation='relu')(x)
        x = layers.MaxPool1D(10)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(50, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)

        self._model = keras.Model(
            inputs=inputs, outputs=outputs, name='fakenews_model')

        self._model.summary(print_fn=logging.info)

        logging.info('compiling the model')
        self._model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                            # Loss function to minimize
                            loss=keras.losses.BinaryCrossentropy(),
                            # List of metrics to monitor
                            metrics=[keras.metrics.BinaryAccuracy()])

    def fit(self, x, y):
        self._model.fit(x, y, epochs=30)

    def predict(self, X):
        return self._model.predict(X)


def loadData():
    logging.info('loading data')

    cache_dir = data_dir / 'cache'
    cached_train = cache_dir / 'train.pkl'
    cached_test = cache_dir / 'test.pkl'

    # read from the cache if data exists
    if os.path.exists(cached_test) and os.path.exists(cached_train):
        logging.info('Read data from cache')
        data_train, data_test = pd.read_pickle(
            cached_train), pd.read_pickle(cached_test)
        return data_train, data_test

    data_train = pd.read_csv(data_dir / 'train.csv', keep_default_na=False)
    data_test = pd.read_csv(data_dir / 'test.csv', keep_default_na=False)

    logging.info('Preprocessing data')

    data_train, data_test = preprocess(data_train, data_test)

    logging.info('Writting data to cache')
    # write to cache
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    data_train.to_pickle(cached_train)
    data_test.to_pickle(cached_test)

    return data_train, data_test


if __name__ == "__main__":
    init_logger()

    data_train, data_test = loadData()

    X_train = np.array(data_train['text'].to_list())
    T_train = np.array(data_train['label'])
    X_test = np.array(data_test['text'].to_list())

    logging.info(
        f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}')

    # test with title first
    cnn = CNN(X_train.shape[1])

    cnn.fit(X_train, T_train)
