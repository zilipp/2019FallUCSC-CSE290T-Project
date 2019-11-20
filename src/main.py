import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# self defined module
from src.utils.utils import init_logger, split_csv


# some config values
_user_logs_file = '..\\out\\logs\\user_logs\\logs.txt'  # User logging directory.
_data_dir = '..\\data'  # Data directory.
_num_folds = 10  # Number of folds(k) used in cross validation.'
_eval_file = 5  # evaluation file in k-fold.
_num_gpus = 2  # Number of GPUs used.

_vocab_size = 50000  # Number of unique words
_seq_length = 5000  # Max length of sequence

if _num_gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def load_data():
    # Check k-fold directory
    logging.info('Checking k-fold files')
    kfold_dir = os.path.join(_data_dir, 'kfold')
    if not os.path.exists(kfold_dir):
        os.mkdir(kfold_dir)
    if not os.listdir(kfold_dir):
        split_csv(os.path.join(_data_dir, 'train.csv'), kfold_dir, _num_folds, 'StratifiedShuffleSplit')

    # Load dataset
    logging.info('Loading train/test dataset')
    kfold_df = [pd.read_csv(os.path.join(kfold_dir, '{}.csv'.format(i))) for i in range(_num_folds)]
    test_df = pd.read_csv(os.path.join(_data_dir, 'test.csv'))
    logging.info('K-fold shape : {0} Test shape : {1}'.format(kfold_df[0].shape, test_df.shape))

    # Split to train and val
    logging.info('Splitting to train and val')
    val_df = kfold_df[_eval_file]
    del kfold_df[_eval_file]
    train_df = pd.concat(kfold_df)
    logging.info('Train shape : {0} Eval shape : {1}'.format(train_df.shape, val_df.shape))

    # Assign values
    logging.info('Assigning values')
    train_x = train_df['text'].fillna('_na_').values
    val_x = val_df['text'].fillna('_na_').values
    test_x = test_df['text'].fillna('_na_').values
    test_id = test_df['id']

    # Get the target values
    # id, title, author, text, label
    logging.info('Get the target values')
    train_y = train_df['label'].values
    val_y = val_df['label'].values

    # Release memory
    del train_df, val_df, test_df

    return train_x, train_y, val_x, val_y, test_x, test_id


def preprocess(train_x, val_x, test_x):
    # Tokenize the sentences
    logging.info('Tokenizing sentences')
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=_vocab_size)
    tokenizer.fit_on_texts(list(train_x))
    train_x = tokenizer.texts_to_sequences(train_x)
    val_x = tokenizer.texts_to_sequences(val_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    # Pad the sentences
    logging.info('Padding sentences')
    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=_seq_length)
    val_x = keras.preprocessing.sequence.pad_sequences(val_x, maxlen=_seq_length)
    test_x = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=_seq_length)

    return train_x, val_x, test_x


def main():
    init_logger(_user_logs_file)
    train_x, train_y, val_x, val_y, test_x, test_id = load_data()
    train_x, val_x, test_x = preprocess(train_x, val_x, test_x)

    logging.info('Done!')


if __name__ == '__main__':
    main()
