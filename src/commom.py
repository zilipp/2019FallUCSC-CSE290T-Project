import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# 1.read csv
train = pd.read_csv('../src/train.csv')
test = pd.read_csv('../src/test.csv')
train_df = train.copy()
test_df = test.copy()

train_df = train_df.set_index('id', drop = True)

# 2. missing value
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

# 3. outliers
train_df = train_df.drop(train_df['text'][train_df['length'] < 10].index, axis = 0)

# common parameters
max_len = 510  # max_len of a sentence(median length of all text)
embed_size = 100  # each word embedded to 300 dimension
max_features = 10000  # count of vocabulary words


# 4.tokenize and padding
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts=train_df['text'])
X = tokenizer.texts_to_sequences(texts=train_df['text'])
X = pad_sequences(sequences=X, maxlen=max_len, padding='pre')
y = train_df['label'].values

# 5. split to train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2019)

