from os import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from preprocess import loadData, get_train_test_set
import logging
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import init_logger, cache_dir

_vocab_size = 10000


class CNN:
    _conv_layer = 'conv'
    _cache_path = cache_dir / 'cnn.h5'

    @classmethod
    def from_cache(cls):
        return CNN(model=keras.models.load_model(cls._cache_path))

    def __init__(self, feature_size=0, vocab_size=0, model=None):
        if (model):
            self._model = model
            return

        if feature_size == 0 or vocab_size == 0:
            raise Exception('feature_size or vocab_size cant be 0')

        self._feature_size = feature_size

        inputs = keras.Input(shape=(feature_size, ))
        # The embedding layer is to use higher dimension real vectors to represent
        # words.
        x = layers.Embedding(vocab_size, 10)(inputs)
        x = layers.Conv1D(50, kernel_size=5, activation='relu')(x)
        x = layers.MaxPool1D(5)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(10, activation='relu')(x)

        outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)

        self._model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='fakenews_model'
        )

        self._model.summary(print_fn=logging.info)

        logging.info('compiling the model')
        self._model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                            # Loss function to minimize
                            loss=keras.losses.BinaryCrossentropy(),
                            # List of metrics to monitor
                            metrics=[keras.metrics.BinaryAccuracy()])

    def fit(self, X, T, epochs=5):
        self._model.fit(X, T, epochs=epochs)

        # save the model
        self._model.save(str(self._cache_path))

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, T):
        logging.info('Evaluating the model with test set')
        logging.info('\n' + tabulate((self._model.test_on_batch(X, T),),
                                     headers=self._model.metrics_names))


if __name__ == "__main__":
    init_logger()

    data_train = loadData('train.csv', ('title', 'text'),
                          vocab_size=_vocab_size)
    X_train, T_train, X_test, T_test = get_train_test_set(data_train)

    logging.info(
        f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}, T_test {T_test.shape}')

    # check from the cache

    # test with title first
    if os.path.exists(cache_dir / CNN._cache_path):
        cnn = CNN.from_cache()
    else:
        cnn = CNN(X_train.shape[1], vocab_size=_vocab_size)
        cnn.fit(X_train, T_train)

    cnn.evaluate(X_test, T_test)
