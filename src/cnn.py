import numpy as np
import tensorflow as ts
import tensorflow.keras as keras
from tensorflow.keras import layers
import logging


class CNN:
    _model = None
    _feature_size = 0

    def __init__(self, feature_size):
        self._feature_size = feature_size

        inputs = keras.Input(shape=(feature_size, ))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Conv1D(30, kernel_size=5, activation='relu')(x)
        outputs = layers.Dense(1, activation='softmax', name="predictions")(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name='fakenews_model')

        self._model.summary(print_fn=logging.info)

        logging.info('compiling the model')
        self._model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                    # Loss function to minimize
                    loss=keras.losses.BinaryCrossentropy(),
                    # List of metrics to monitor
                    metrics=[keras.metrics.BinaryCrossentropy(), keras.metrics.BinaryAccuracy()])

    def fit(self, x, y, x_val, y_val):
        self._model.fit(x, y, epochs = 3, validation_data=(x_val, y_val))

    def predict(self, X):
        return self._model.predict(X)
