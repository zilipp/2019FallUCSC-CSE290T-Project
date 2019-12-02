from os import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import data_dir, init_logger
from typing import List

_vocab_size = 10000


class CNN:
    _model = None
    _feature_size = 0

    def __init__(self, feature_size, vocab_size):
        self._feature_size = feature_size

        inputs = keras.Input(shape=(feature_size, ))
        # The embedding layer is to use higher dimension real vectors to represent
        # words.
        x = layers.Embedding(vocab_size, 1)(inputs)
        x = layers.Conv1D(50, kernel_size=5, activation='relu')(x)
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

    def fit(self, X, T, epochs=5):
        self._model.fit(X, T, epochs=epochs)

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, T):
        logging.info('Evaluating the model with test set')
        logging.info('\n' + tabulate((self._model.test_on_batch(X, T),),
                                     headers=self._model.metrics_names))


def loadData(filename, cols=List[str], tokenizer_name=None):
    """Load file into vector data with vocabulary file name"""
    [filename, ext] = os.path.splitext(filename)
    if ext != '.csv':
        raise Exception('Only support .csv files')

    logging.info(f'loading data from {filename}')

    cache_dir = data_dir / 'cache'
    tokenizer_name = tokenizer_name if tokenizer_name else filename
    cached_filename = cache_dir / f'{filename}.pkl'

    # read from the cache if data exists
    if os.path.exists(cached_filename):
        logging.info('Read data from cache')
        return pd.read_pickle(cached_filename)

    data = pd.read_csv(data_dir / f'{filename}.csv', keep_default_na=False)

    # write to cache
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    logging.info('Preprocessing data')
    for col in cols:
        preproc = Preprocessor(cache_path=cache_dir /
                               f'{tokenizer_name}_{col}.json', num_words=_vocab_size)
        preproc.fit(data[col])
        data[col] = preproc.transform(data[col])
        preproc.save()

    logging.info('Writting data to cache')

    data.to_pickle(cached_filename)

    return data


if __name__ == "__main__":
    init_logger()

    data_train = loadData('train.csv', ('title', 'text'))

    data_train, data_test = train_test_split(
        data_train, test_size=0.2, stratify=data_train['label'])

    X_train = np.array(data_train['text'].to_list())
    T_train = np.array(data_train['label'])
    X_test = np.array(data_test['text'].to_list())
    T_test = np.array(data_test['label'])

    logging.info(
        f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}, T_test {T_test.shape}')

    # check from the cache

    # test with title first
    cnn = CNN(X_train.shape[1], vocab_size=_vocab_size)
    cnn.fit(X_train, T_train)
    cnn.evaluate(X_test, T_test)
