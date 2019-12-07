# common.py can be directly used in kaggle platform kernels
# all these code in the file used to know the basic information of dataset
#


# importing necessary libraries
import pandas as pd
import tensorflow as tf
import os
import re
import numpy as np
from string import punctuation
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# 1.read csv
train = pd.read_csv('../src/train.csv')
test = pd.read_csv('../src/test.csv')
train_df = train.copy()
test_df = test.copy()


# 2. visualization - word cloud
sincere_df = train.loc[train.label == 0]
print(sincere_df.head())
insincere_df = train.loc[train.label == 1]
print(insincere_df.head())

sincere_text_array = np.array(sincere_df['text'])
insincere_text_array = np.array(insincere_df['text'])
sincere_text = ''.join(sincere_text_array)
insincere_text = ''.join(str(e) for e in insincere_text_array)
print("convert to text finished")

# sincere
wordcloud = WordCloud(background_color="white").generate(sincere_text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# insincere
wordcloud = WordCloud(background_color="white").generate(insincere_text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


train_df = train_df.set_index('id', drop = True)
# 3. missing value
print(train_df.isnull().sum())
# title      558
# author    1957
# text        39
# label        0
train_df[['title', 'author']] = train_df[['title', 'author']].fillna(value='Missing')
train_df = train_df.dropna()
print(train_df.isnull().sum())
# title     0
# author    0
# text      0
# label     0

# 4. outliers
train_df = train_df.drop(train_df['text'][train_df['length'] < 10].index, axis = 0)

# common parameters
max_len = 510  # max_len of a sentence(median length of all text)
embed_size = 100  # each word embedded to 300 dimension
max_features = 10000  # count of vocabulary words


# 5.tokenize and padding
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts=train_df['text'])
X = tokenizer.texts_to_sequences(texts=train_df['text'])
X = pad_sequences(sequences=X, maxlen=max_len, padding='pre')
y = train_df['label'].values

# 6. split to train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2019)

