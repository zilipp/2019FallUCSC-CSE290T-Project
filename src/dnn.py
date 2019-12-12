import os
import pandas as pd
import logging


from tensorflow_estimator.python.estimator.canned.dnn import DNNClassifier
from utils import init_logger
import tensorflow as tf
from tensorflow import keras
from preprocess import loadData, get_train_test_set

init_logger()
_vocab_size = 10000
_seq_length = 150
_model_dir = "./tmp/dnn_model"

# data preprocessing
data_train = loadData('train.csv', ('title', 'text'), vocab_size=_vocab_size)
X_train, T_train, X_test, T_test = get_train_test_set(data_train)
# logging.info(f'X_train {X_train.shape}, T_train {T_train.shape}, X_test {X_test.shape}, T_test {T_test.shape}')
feature_size = X_train.shape[1]

# build model
model = keras.models.Sequential()
model.add(keras.layers.Embedding(_vocab_size, 20, input_shape=(feature_size,)))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.GlobalMaxPool1D())
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])

print("Starting training ")
num_epochs = 10
h = model.fit(X_train, T_train, batch_size=50, epochs=num_epochs, verbose=0)
print("Training finished \n")

logging.info("Start evaluating")
model.fit(X_train, T_train, epochs=10)
model.evaluate(X_test, T_test)
logging.info(model.summary())
