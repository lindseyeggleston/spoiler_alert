'''
This script takes 2 additional arguments and runs like so:
    $ python keras_rnn.py text name
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
BATCH_SIZE = 500
LEARNING_RATE = 0.01
INTERNAL_SIZE = 512
VOCAB_SIZE = 8000 # number of words in vocabulary
STEP = 7

def _build_rnn():
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

def _vectorize_text(text, vocab_dict):
    print('Vectorizing text...')

    sequences = []
    next_words = []

    for i in range(0, len(text)-SEQ_LENGTH, STEP):
        sequences.append(text[i: i + SEQ_LENGTH])
        next_words.append(text[i + SEQ_LENGTH])

    vec = np.vectorize(lambda x: vocab_dict[x])
    X = vec(np.array(sequences))
    y = vec(np.array(next_words))

    Xo = np.zeros((X.shape[0], X.shape[1], VOCAB_SIZE), dtype=np.int8)
    yo = np.zeros((y.shape[0], VOCAB_SIZE), dtype=np.int8)

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            Xo[i, j, val] = 1
        yo[i, y[i]] = 1

    return Xo, yo

def train_rnn(text, name):
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

    save_as = '../model/{0}.h5'.format(name)
    rnn.save_weights(save_as, overwrite=True)
    rnn.save(save_as, overwrite=True)



if __name__ == '__main__':
    _, text, name = argv
    train_rnn(text, name)
