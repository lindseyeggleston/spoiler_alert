import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
import data_processing as dp
import os

N_EPOCHS = 10
SEQ_LENGTH = 30 # length of input sequence
BATCH_SIZE = 300
INTERNAL_SIZE = 512 # number of nodes in each hidden layer
DROPOUT = 0.2 # percent of data to dropout at each layer
LEARNING_RATE = 0.001


def _build_rnn(n_chars):

    # Construct model with LSTM architecture
    print('Building model...')
    rnn = Sequential([
        Embedding(n_chars, n_chars),
        LSTM(INTERNAL_SIZE, input_shape=(None, SEQ_LENGTH, n_chars)),
        Dropout(DROPOUT),
        Dense(n_chars, activation='softmax')
    ])

    # Learning rate optimization (either RMSprop or Adam)
    optimizer = Adam(lr=LEARNING_RATE)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return rnn


def train_rnn(text):

    # Dictionaries
    char_index = dp.count_characters
    n_chars = len(char_index)
    index_char = dict(list(zip(char_index.values(), char_index.keys())))

    rnn = _build_rnn(n_chars)

    # Mini-batch stocastic
    print('Training...')
    for X, y, epoch in dp.create_minibatch(text, BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS):
        rnn.train_on_batch(X, y)

    # Save trained model to pickle file
    title = raw_input('Enter a filename for this model: ')
    if os.path.isfile('../model/{0}.pkl'.format(title)):
        print('This filename already exist')
        title = raw_input('Enter a new filename: ')
    filename = '../model/{0}.pkl'.format(title)
    with open(filename, 'w') as f:
        pickle.dump(rnn, f)
        print('Your model has been pickled')

    return None
