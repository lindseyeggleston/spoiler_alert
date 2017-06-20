'''
This script takes 2 additional arguments and runs like so:
    $ python keras_rnn.py filepath save_as
'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
from sys import argv
import data_processing as dp

# model hyper parameters
N_EPOCHS = 10
SEQ_LENGTH = 30
DROPOUT = 0.2
BATCH_SIZE = 300
LEARNING_RATE = 0.001
INTERNAL_SIZE = 100 # number of nodes in each hidden layer
VOCAB_SIZE = 5000 # number of words in vocabulary
STEP = 3

def _build_rnn():
    '''
    Builds RNN model framework according to above hyperparameter specifications
    '''
    print('Building model...')
    rnn = Sequential([      # linear stack of layers
        LSTM(INTERNAL_SIZE, return_sequences=True, input_shape=(SEQ_LENGTH, VOCAB_SIZE)), # return_sequences = True b/c many-to-many model
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE),
        Dropout(DROPOUT),
        Dense(VOCAB_SIZE),
        Activation('softmax')])

    optimizer = Adam(lr=LEARNING_RATE)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return rnn

def _vectorize_text(text, word_indices, seq_length=SEQ_LENGTH, step=STEP):
    '''
    Vectorizes text string into sparse index matric

    Parameters
    ----------
    text: STR - text corpus/corpora
    word_indices: DICT - a dictionary of vocabulary with words as the keys and
        their indices as the values
    seq_length: INT - length of input sequence
    step: INT - step size

    Returns
    -------
    input matric, Xo, and its corresponding target values, yo
    '''
    print('Vectorizing text...')

    sequences = []
    next_words = []

    for i in range(0, len(text)-seq_length, step):
        sequences.append(text[i: i + seq_length])
        next_words.append(text[i + seq_length])

    vec = np.vectorize(lambda x: word_indices[x])
    X = vec(np.array(sequences))
    y = vec(np.array(next_words))

    Xo = np.zeros((X.shape[0], X.shape[1], len(word_indices)), dtype=np.int8)
    yo = np.zeros((y.shape[0], len(word_indices)), dtype=np.int8)

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            Xo[i, j, val] = 1
        yo[i, y[i]] = 1

    return Xo, yo

def train_rnn(text, save_as):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameters
    ----------
    text: STR - text corpus/corpora
    model_title: STR - the filename for the saved model

    Returns
    -------
    None
    '''
    tokens = dp.tokenize_text(text)
    tokens, word_indices, unknown_tokens = dp.text_to_vocab(tokens, vocab_size=VOCAB_SIZE)
    indices_word = dict((v,k) for k,v in word_indices.items())

    X, y = _vectorize_text(tokens, word_indices)

    rnn = _build_rnn()

    print('Training...')
    rnn.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

    name = '../model/{0}.h5'.format(save_as)
    rnn.save_weights(name, overwrite=True)
    rnn.save(name, overwrite=True)
    return rnn



if __name__ == '__main__':
    _, filepath, save_as = argv
    text = dp.file_to_text(filepath)
    train_rnn(text, save_as)
