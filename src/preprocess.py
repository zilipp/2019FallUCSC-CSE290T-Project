import os
import tensorflow as ts
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import logging


class Preprocessor:
    _tk = None
    _cache_path = None

    def __init__(self, cache_path=None, **extra):
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._tk = tokenizer_from_json(f.read())
        else:
            self._tk = Tokenizer(lower=True, **extra)

        self._cache_path = cache_path

    def fit(self, data):
        self._tk.fit_on_texts(data)

    def save(self):
        filename = self._cache_path
        with open(filename, 'w') as f:
            f.write(self._tk.to_json())

    def transform(self, data: pd.Series, truncate='median'):
        """Transform a list of Series of texts into a list of Series of vectors"""
        seq = self._tk.texts_to_sequences(data)

        if truncate == 'median':
            text_len = int(np.median([len(vec) for vec in seq]))
        else:
            text_len = int(truncate)

        logging.info(f'Transforming texts into vectors with {text_len} size')

        return pd.Series(pad_sequences(seq, padding='post', maxlen=text_len).tolist())
