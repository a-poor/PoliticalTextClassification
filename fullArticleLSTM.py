"""


"""

###### IMPORTS ######

import sqlite3
import pickle
import re
import random
from time import strftime, localtime

import numpy as np

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


###### SET SOME SETTINGS ######

MAX_ARTICLE_LENGTH = 10000
PAD_VALUE = 0

###### CONNECTING TO THE ENCODER DB ######

db = sqlite3.connect('db/text_encoder.db')
c = db.cursor()
c.execute('CREATE TABLE IF NOT EXISTS "TextEncoder"(\n\t"word" TEXT,\n\t"n" INTEGER,\n\tPRIMARY KEY("word")\n);')
del c


###### HELPER FUNCTIONS ######

def get_max_key():
    c = db.cursor()
    c.execute('SELECT MAX(n) FROM TextEncoder;')
    return c.fetchone()[0]

def add_word_encoding(word, key):
    c = db.cursor()
    c.execute('INSERT INTO TextEncoder (word, n) VALUES (?, ?);', (word, key))
    assert c.rowcount > 0, "ERROR Inserting key %10s as word: %20s" % (key, word)
    db.commit()
    return

def get_word_key(word):
    c = db.cursor()
    c.execute('SELECT n FROM TextEncoder WHERE word = ?;', (word,))
    response = c.fetchone()
    if response is not None:
        return response[0]
    else:
        key = get_max_key() + 1
        add_word_encoding(word, key)
        print('Adding word "%20s" as n: %10i' % (word, key))
        return key

def get_vocab_length():
    c = db.cursor()
    c.execute('SELECT COUNT(*) FROM TextEncoder;')
    return c.fetchone()[0]

def numberize_article(article):
    tokens = text_to_word_sequence(article)
    num_article = []
    for token in tokens:
        num_article.append(get_word_key(token))
    return np.array(num_article)

def get_nums_word(n):
    c = db.cursor()
    c.execute('SELECT word FROM TextEncoder WHERE n = ?;', (n,))
    word = c.fetchone()
    if word is None:
        return ''
    else:
        return word[0]


###### USER FUNCTIONS ######

def remake_article(num_article):
    text_array = [get_nums_word(n) for n in num_article]
    raw_text = ' '.join(text_array)
    return re.sub(r'\s+', ' ', raw_text)


def parse_articles(articles, max_length=MAX_ARTICLE_LENGTH, min_length=20):
    numerified_articles = []
    removed_articles = set()
    for i, a in enumerate(articles):
        numberized = numberize_article(a)
        if len(numberized) >= min_length:
            numerified_articles.append(numberized)
        else:
            removed_articles.add(i)
            print('Removing article: %6i for being too short. Length: %3i' % (i, len(numberized)))
    numerified_articles = np.array(numerified_articles)
    return pad_sequences(numerified_articles, maxlen=max_length, dtype='float32', value=0), removed_articles


def train_test_split(data_in, data_out, train_pct=0.75, validate=True, val_pct_of_test=0.33):
    all_data = list(zip(data_in, data_out))
    random.shuffle(all_data)
    shuffled_in, shuffled_out = zip(*all_data)
    data_length = len(all_data)
    if validate:
        test_split = int(data_length * train_pct)
        val_split = test_split + int((data_length - test_split) * (1 - val_pct_of_test))
        
        train_in = shuffled_in[:test_split]
        test_in = shuffled_in[test_split:val_split]
        val_in = shuffled_in[val_split:]
        
        train_out = shuffled_out[:test_split]
        test_out = shuffled_out[test_split:val_split]
        val_out = shuffled_out[val_split:]
        
        return (np.array(train_in), np.array(train_out)), (np.array(val_in), np.array(val_out)), (np.array(test_in), np.array(test_out))
    else:
        test_split = int(data_length * train_pct)
        train_in = shuffled_in[:test_split]
        train_out = shuffled_out[:test_split]
        test_in = shuffled_in[test_split:]
        test_out = shuffled_out[test_split:]
        return (np.array(train_in), np.array(train_out)), (np.array(test_in), np.array(test_out))


def create_model(encoding_vector_size=100, n_filters=32, kernel_size=3, pool_size=2, add_dropout=True, dropout_pct=0.2):
    model = Sequential()
    model.add(Embedding(
        input_dim=get_vocab_length(),
        output_dim=encoding_vector_size,
        input_length=MAX_ARTICLE_LENGTH
    ))
    if add_dropout:
        model.add(Dropout(dropout_pct))
    model.add(Conv1D(
        filters=n_filters, 
        kernel_size=kernel_size, 
        padding='same', 
        activation='relu'
    ))
    model.add(MaxPooling1D(
        pool_size=pool_size
    ))
    if add_dropout:
        model.add(Dropout(dropout_pct))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    return model

def gen_filename():
    return 'savedModels/article_lstm_model_%s.h5' % strftime('%y%m%d.%H%M%S', localtime())

def get_data_from_db():
    adb = sqlite3.connect('db/articles.db')
    c = adb.cursor()
    c.execute('SELECT text, right_score FROM SiteScrape;')
    articles, labels = zip(*c.fetchall())
    return articles, np.array(labels).astype('float32')

def save_data(articles, labels):
    data = {
        'articles': articles,
        'labels': labels
    }
    with open('processedAndLabeledArticles.pickle', 'xb') as f:
        pickle.dump(data, f)
    return

def get_data_from_pickle():
    with open('processedAndLabeledArticles.pickle', 'rb') as f:
        data = pickle.load(f)
    articles = data['articles']
    labels   = data['labels']
    return articles, labels

###### START OF DOING STUFF ######

### Get the data from the db (if there are any updates)

# articles, labels = get_data_from_db()
# processed_articles, removed_articles = parse_articles(articles)
# labels = np.array([l for i, l in enumerate(labels) if i not in removed_articles])
# ### The save it out again
# save_data(processed_articles, labels)

### Otherwise, load it from here…
processed_articles, labels = get_data_from_pickle()


# Split up the data
print('Splitting the data…')
(train_in, train_out), (val_in, val_out), (test_in, test_out) = train_test_split(processed_articles, labels)


# Build the model
print('Building the model…')
model = create_model()
# model = load_model('')

callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=2
        ),
    ModelCheckpoint(
        filepath=gen_filename(), 
        monitor='val_acc', 
        save_best_only=True
        )
    ]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
    )


print('Model summary…')
print(model.summary())


print('Training…')
hist = model.fit(
    train_in,
    train_out,
    epochs=3,
    batch_size=64,
    validation_data=(val_in, val_out),
    callbacks=callbacks
                )


print('Evaluating…')
eval = model.evaluate(test_in, test_out)
print(eval)
