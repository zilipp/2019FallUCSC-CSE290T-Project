from os import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import logging
from tabulate import tabulate
from utils import data_dir, init_logger

_vocab_size = 10000


class CNN:
    _model = None
    _feature_size = 0

    def __init__(self, feature_size):
        self._feature_size = feature_size

        logging.info(tf.config.experimental.list_physical_devices('GPU'))

        inputs = keras.Input(shape=(feature_size, ))
        x = layers.Embedding(_vocab_size, 10)(inputs)
        x = layers.Conv1D(50, kernel_size=10, activation='relu')(x)
        x = layers.MaxPool1D(5)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(10, activation='relu')(x)

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

    def fit(self, X, T):
        self._model.fit(X, T, epochs=5)

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, T):
        logging.info('Evaluating the model with test set')
        logging.info('\n' + tabulate((self._model.test_on_batch(X, T),),
                                     headers=self._model.metrics_names))


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

    data_train, data_test = preprocess(data_train, data_test, _vocab_size)

    logging.info('Writting data to cache')
    # write to cache
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    data_train.to_pickle(cached_train)
    data_test.to_pickle(cached_test)

    return data_train, data_test


if __name__ == "__main__":
    init_logger()

    data_train, data_unknown = loadData()

    data_train, data_test = train_test_split(data_train, test_size=0.2)

    X_train = np.array(data_train['text'].to_list())
    T_train = np.array(data_train['label'])
    X_test = np.array(data_test['text'].to_list())
    T_test = np.array(data_test['label'])

    logging.info(
        f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}, T_test {T_test.shape}')

    # check from the cache

    # test with title first
    cnn = CNN(X_train.shape[1])
    cnn.fit(X_train, T_train)
    cnn.evaluate(X_test, T_test)
