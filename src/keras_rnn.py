from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import sys
import pickle

# model hyper parameters
epochs = 10
seq_length = 25
n_layers = 3
dropout = 0.2
batch_size = 200
learning_rate = 0.01
internal_size = 128

def train_rnn(raw_data):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameter
    ---------
    raw_data: text corpora
    '''

    words = set(raw_data.strip('\n').split())

    word_indices = dict((word, idx) for idx, word in enumerate(words))
    indices_word = dict((idx, word) for idx, word in enumerate(words))

    # Build a model: see hyper parameters above
    print('Building model...')

    rnn = Sequential()
    for i in range(n_layers-1):
        rnn.add(LSTM(internal_size, return_sequences=True, input_shape=(batch_size, len(words))))
        rnn.add(Dropout(dropout))
    rnn.add(LSTM(internal_size, return_sequences=False))
    rnn.add(Dropout(dropout))
    rnn.add(Dense(len(words)))
    rnn.add(Activation('softmax'))

    optimizer = RMSprop(lr=learning_rate)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

    #
    step = 3
    sentences = []
    next_words = []
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length])
        next_words.append(text[i + seq_length])

    print('Vectorization...')
    X = np.zeros((len(sentences), seq_length, len(words)), dtype=np.bool)
    y = np.zeros((len(sentences), len(words)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            X[i, t, word_indices[word]] = 1
        y[i, word_indices[next_words[i]]] = 1
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    filename = '../model/rnn.pkl'
    with open(filename, 'w') as f:
        pickle.dump(model, f)
        print('Your model has been pickled')
    return None

if __name__ == '__main__':
    with open('../data/SOIF/AGameOfThrones.txt') as f:
        text = f.read()
    train_rnn(text)
