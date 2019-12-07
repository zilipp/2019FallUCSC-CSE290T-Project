from os import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from preprocess import loadData, get_train_test_set, Preprocessor
import logging
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import init_logger, cache_dir, out_dir
import cv2
from yattag import Doc

_vocab_size = 20000


class CNN:
    _conv_layer = 'conv'
    _raw_output_layer = 'raw'
    _cache_path = cache_dir / 'cnn.h5'

    @classmethod
    def from_cache(cls):
        logging.info('getting model from the cache')
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
        x = layers.Embedding(vocab_size, 20)(inputs)
        # This is a CNN that has 50 filters with a 5x5 window
        x = layers.Conv1D(
            20, kernel_size=3, activation='relu', name=self._conv_layer)(x)
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(2, name=self._raw_output_layer)(x)
        outputs = keras.activations.softmax(x)

        self._model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='fakenews_model'
        )

        self._model.summary(print_fn=logging.info)

        logging.info('compiling the model')
        self._model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                            # Loss function to minimize
                            loss=keras.losses.CategoricalCrossentropy(),
                            # List of metrics to monitor
                            metrics=[keras.metrics.CategoricalAccuracy()])

    def fit(self, X, T, X_val, T_val, epochs=5):
        history = self._model.fit(
            X, T, validation_data=(X_val, T_val), epochs=epochs)

        # save the model
        self._model.save(str(self._cache_path))
        return history

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, T):
        logging.info('Evaluating the model with test set')
        logging.info('\n' + tabulate((self._model.test_on_batch(X, T),),
                                     headers=self._model.metrics_names))

    def predict_to_classes(self, X):
        prediction = self.predict(X)

        return (prediction > 0.5).astype(int)

    def get_cam(self, X, label=1):
        model = self._model
        grad_model = keras.Model(inputs=model.inputs,
                                 outputs=[model.get_layer(self._conv_layer).output,
                                          model.get_layer(
                                              self._raw_output_layer).output,
                                          model.output
                                          ])

        grad_model.summary()

        with tf.GradientTape() as tape:
            feature_map, raw_prediction, prediction = grad_model(X)
            if label == 1:
                loss = raw_prediction[:, 0]
            else:
                loss = raw_prediction[:, 1]

        grads = tape.gradient(loss, feature_map)
        grads_mean = tf.reduce_mean(grads, 1)
        weights = tf.reshape(grads_mean, grads_mean.shape + (1,))

        # calculate the cam heat map data
        cams = tf.nn.relu(feature_map @ weights).numpy()

        new_cams = []

        for cam in cams:
            cam = cv2.resize(cam, (cam.shape[1], model.input.shape[1]))
            new_cams.append(cam)

        cams = np.array(new_cams)[..., 0]

        return cams


def print_document(X, Y, T, cams):
    doc, tag, text = Doc().tagtext()

    prepro = Preprocessor(cache_path=cache_dir / 'train_text.json')
    X_text = prepro.to_text(X_sample)

    with tag('html'):
        with tag('body', style="width: 900px;"):
            for i, p in enumerate(X_text):
                cam = cams[i]
                # normalize cam
                heatmap = cam / np.ptp(cam)
                color_map = cv2.applyColorMap(
                    np.uint8(255 * heatmap), cv2.COLORMAP_AUTUMN)
                with tag('div'):
                    with tag('p'):
                        words = p.split(' ')
                        for j, word in enumerate(words):
                            color = color_map[j][0]
                            with tag('span', style=f'background: rgba({color[2]}, {color[1]}, {color[0]}, {heatmap[j]});', title=int(X_test[i, j])):
                                text(word + ' ')
                    with tag('p'):
                        text(
                            f'Pred: {Y[i]}, Label: {T[i]}')
                    doc.stag('hr')

    with open(out_dir / 'out.html', 'w') as f:
        f.write(doc.getvalue())


if __name__ == "__main__":
    init_logger()

    data_train, _ = loadData('train.csv', ('title', 'text'),
                             vocab_size=_vocab_size)
    X_train, T_train, X_test, T_test = get_train_test_set(data_train)

    logging.info(
        f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}, T_test {T_test.shape}')

    T_train = np.hstack((T_train.reshape(-1, 1), (1 - T_train).reshape(-1, 1)))
    T_test = np.hstack((T_test.reshape(-1, 1), (1 - T_test).reshape(-1, 1)))

    # check from the cache

    # test with title first
    if os.path.exists(cache_dir / CNN._cache_path):
        cnn = CNN.from_cache()
    else:
        cnn = CNN(X_train.shape[1], vocab_size=_vocab_size)
        history = cnn.fit(X_train, T_train, X_test, T_test, epochs=20)
        print(history.history)

    cnn.evaluate(X_test, T_test)

    T_fake = (T_test[:, 0] == 1)
    X_sample, T_sample = X_test[T_fake, :][60:90], T_test[T_fake, :][60:90]

    prediction = cnn.predict_to_classes(X_sample)
    cams = cnn.get_cam(X_sample, 1)

    print_document(X_sample, prediction, T_sample, cams)
