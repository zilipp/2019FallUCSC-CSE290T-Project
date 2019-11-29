import tensorflow as ts
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging


class Preprocessor:
    _tk = None

    def __init__(self, **extra):
        self._tk = Tokenizer(lower=True, **extra)

    def fit(self, data):
        self._tk.fit_on_texts(data)

    def transform(self, data):
        return self._tk.texts_to_sequences(data)


def preprocess(data_train, data_test):
    proc_title = Preprocessor()
    proc_title.fit(data_train['title'])
    proc_title.fit(data_test['title'])

    proc_text = Preprocessor()
    proc_text.fit(data_train['text'])
    proc_text.fit(data_test['text'])

    def transform_col(preprocessor: Preprocessor, col: str):
        col_train = preprocessor.transform(data_train[col])
        col_test = preprocessor.transform(data_test[col])

        median = int(
            np.median(list(map(len, col_train + col_test))))

        logging.info(f'Transforming {col} into vectors with {median} size')

        data_train[col] = pd.Series(pad_sequences(
            col_train, padding='post', maxlen=median).tolist())
        data_test[col] = pd.Series(pad_sequences(
            col_test, padding='post', maxlen=median).tolist())

    transform_col(proc_title, 'title')
    transform_col(proc_text, 'text')

    return data_train, data_test
