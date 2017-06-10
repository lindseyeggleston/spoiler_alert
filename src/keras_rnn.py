from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import sys
import pickle

# model hyper parameters
num_epochs = 10
seq_length = 30
n_layers = 3
dropout = 0.2
batch_size = 500
learning_rate = 0.01
internal_size = 512

def train_rnn(raw_data):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameter
    ---------
    raw_data: text corpora
    '''

    text = raw_data.split(' ')
    words = set(text)
    vocab_length = len(words)

    word_indices = dict((word, idx) for idx, word in enumerate(words))
    indices_word = dict((idx, word) for idx, word in enumerate(words))

    # Build a model: see hyper parameters above
    print('Building model...')

    rnn = Sequential()
    for i in range(n_layers-1):
        rnn.add(LSTM(internal_size, return_sequences=True, input_shape=(seq_length, vocab_length)))
        rnn.add(Dropout(dropout))
    rnn.add(LSTM(internal_size, return_sequences=False))
    rnn.add(Dropout(dropout))
    rnn.add(Dense(vocab_length))
    rnn.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # Mini-batch stocastic
    print('Training...')
    for X, y, epoch in create_minibatch(text, batch_size, seq_length, num_epochs):
        rnn.fit(X, y, batch_size=batch_size, verbose=1)

    # Save trained model to pickle file
    filename = '../model/rnn.pkl'
    with open(filename, 'w') as f:
        pickle.dump(model, f)
        print('Your model has been pickled')
    return None

if __name__ == '__main__':
    with open('../data/SOIF/AGameOfThrones.txt') as f:
        text = f.read()
    train_rnn(text)
