import os
import tensorflow as ts
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import data_dir, cache_dir
from typing import List, Union

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

    def transform(self, data: pd.Series, truncate: Union[str, int] = 'median'):
        """Transform a list of Series of texts into a list of Series of vectors"""
        seq = self._tk.texts_to_sequences(data)

        lens = [len(vec) for vec in seq]
        logging.info(
            f'median {np.median(lens)}, mean {np.mean(lens)}, max {np.max(lens)}, min {np.min(lens)}')

        if truncate == 'median':
            text_len = int(np.median(lens))
        else:
            text_len = truncate

        logging.info(f'Transforming texts into vectors with {text_len} size')

        return pd.Series(pad_sequences(seq, padding='post', maxlen=text_len).tolist())

    def to_text(self, data):
        """Transform a vector back to text

        Arguments:
            data {list} -- ndarray or pd.Series
        """

        return self._tk.sequences_to_texts(data)


def loadData(filename, cols: List[str], tokenizer_name=None, vocab_size=10000):
    """Load data from csv files

    Arguments:
        filename {string} -- the file name in data directory

    Keyword Arguments:
        cols {List[str]} -- the columns that needed to be preprocessed
        tokenizer_name {str} -- Name of the tokenizer for reuse (default: {None})
        vocab_size {int} -- vocabulary size for Tokenizer (default: {10000})

    Returns:
        [pandas.DataFrame] -- the proccessed DataFrame
    """

    [filename, ext] = os.path.splitext(filename)
    if ext != '.csv':
        raise Exception('Only support .csv files')

    logging.info(f'loading data from {filename}')
    data = pd.read_csv(data_dir / f'{filename}.csv', keep_default_na=False)

    tokenizer_name = tokenizer_name if tokenizer_name else filename
    cached_filename = cache_dir / f'{filename}.pkl'

    # read from the cache if data exists
    if os.path.exists(cached_filename):
        logging.info('Read data from cache')
        return pd.read_pickle(cached_filename), data

    # write to cache
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    logging.info('Preprocessing data')
    original_data = data
    data = data.copy()
    for col in cols:
        preproc = Preprocessor(cache_path=cache_dir /
                               f'{tokenizer_name}_{col}.json', num_words=vocab_size)
        preproc.fit(data[col])
        data[col] = preproc.transform(data[col])
        preproc.save()

    logging.info('Writting data to cache')

    data.to_pickle(cached_filename)

    return data, original_data


def get_train_test_set(data: pd.DataFrame, col='text', test_size=0.2, min_seq=10):
    """Turn the DataFrame into train and test sets

    Arguments:
        data {DataFrame} -- The DataFrame to work with

    Keyword Arguments:
        col {str} -- The target column in the DataFrame (default: {'text'})
        test_size {float} -- Size of the test set, [0, 1] (default: {0.2})
        min_seq {int} -- the minium sequence. Vectors shorter than this will be dropped (default: {50})

    Returns:
        (ndarray, ndarray, ndarray, ndarray) -- X_train, T_train, X_test, T_test
    """

    data = data[[col, 'label']]
    data = data.drop(data[np.fromiter(map(lambda i: len(np.trim_zeros(i)), data[col]), int)
                          < min_seq].index, axis=0)

    data_train, data_test = train_test_split(
        data, test_size=test_size, stratify=data['label'])

    X_train = np.array(data_train[col].to_list())
    T_train = np.array(data_train['label'])
    X_test = np.array(data_test[col].to_list())
    T_test = np.array(data_test['label'])

    return X_train, T_train, X_test, T_test
