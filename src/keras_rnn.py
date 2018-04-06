'''
This script takes 2 additional arguments and runs like so:
    $ python keras_rnn.py filepath save_as
'''

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
from sys import argv
import data_processing as dp
import pickle

# model hyper parameters
N_EPOCHS = 10
SEQ_LENGTH = 30
DROPOUT = 0.2
BATCH_SIZE = 300
LEARNING_RATE = 0.001
INTERNAL_SIZE = 512 # number of nodes in each hidden layer
VOCAB_SIZE = 5000 # number of words in vocabulary
STEP = 3

def _build_rnn():
    '''
    Builds RNN model framework according to above hyperparameter specifications
    '''
    print('Building model...')
    rnn = Sequential([      # linear stack of layers
        Embedding(VOCAB_SIZE, INTERNAL_SIZE, inpute_length=SEQ_LENGTH)
        LSTM(INTERNAL_SIZE, return_sequences=True, input_shape=(SEQ_LENGTH, VOCAB_SIZE)), # return_sequences = True b/c many-to-many model
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE),
        Dropout(DROPOUT),
        Dense(VOCAB_SIZE),
        Activation('softmax')])

    optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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

    return X, y

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
    tokens, word_indices, precedes_unknown_token = dp.text_to_vocab(tokens, vocab_size=VOCAB_SIZE)
    indices_word = dict((v,k) for k,v in word_indices.items())

    X, y = _vectorize_text(tokens, word_indices)

    rnn = _build_rnn()

    print('Training...')
    rnn.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

    name = '../model/{0}'.format(save_as)
    with open(name + '_vocab.pkl', 'wb') as f:
        pickle.dump(word_indices, f)
    with open(name + '_unknown.pkl', 'wb') as f:
        pickle.dump(precedes_unknown_token, f)
    rnn.save_weights(name + '.h5', overwrite=True)
    rnn.save(name, overwrite=True)
    return rnn



if __name__ == '__main__':
    _, filepath, save_as = argv
    text = dp.file_to_text(filepath)
    train_rnn(text, save_as)
