from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import sys
import pickle
import data_processing as dp
from string import punctuation

# model hyper parameters
num_epochs = 10
seq_length = 30
n_layers = 3
dropout = 0.2
batch_size = 500
learning_rate = 0.01
internal_size = 512
vocab_size = 8000 # number of words in vocabulary

def train_rnn(raw_data):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameter
    ---------
    raw_data: (iterable) text corpora
    '''

    text, vocab, unknown_tokens = dp.text_to_vocab(raw_data, vocab_size=vocab_size)
    data = dp.text_to_sequence(text, vocab)

    # Build a model: see hyper parameters above
    print('Building model...')
    rnn = Sequential([
        Embedding(vocab_size, vocab_size, input_length=seq_length),
        LSTM(internal_size, return_sequences=True, input_shape=(None, seq_length, internal_size)),
        Dropout(dropout),
        LSTM(internal_size, return_sequences=True),
        Dropout(dropout),
        LSTM(internal_size, return_sequences=False),
        Dropout(dropout),
        Dense(vocab_size),
        Activation('softmax')])

    optimizer = Adam(lr=learning_rate)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # Mini-batch stocastic
    print('Training...')
    for X, y, epoch in dp.create_minibatch(text, batch_size, seq_length, num_epochs):
        rnn.train_on_batch(X, y)

    # Save trained model to pickle file
    filename = '../model/rnn.pkl'
    with open(filename, 'w') as f:
        pickle.dump(model, f)
        print('Your model has been pickled')
    return None

if __name__ == '__main__':
    with open('../../soif_data/text/book1.txt') as f:
        text = f.read()
        text = ''.join(char for char in text.lower() if char not in punctuation)
        text = text.split(' ')
    train_rnn(text)
