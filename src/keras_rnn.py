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
NUM_EPOCHS = 10
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
        Embedding(VOCAB_SIZE, VOCAB_SIZE, input_length=SEQ_LENGTH),
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

def train_rnn(raw_data):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameter
    ---------
    raw_data: (iterable) text corpora
    '''
    tokens = dp.tokenize_text(text)
    tokens, word_indices, unknown_tokens = dp.text_to_vocab(tokens, vocab_size=VOCAB_SIZE)
    indices_word = dict((v,k) for k,v in word_indices.items())

    sequences = []
    next_words = []

    for i in range(0, len(tokens)-SEQ_LENGTH, STEP):
        sequences.append(tokens[i: i + SEQ_LENGTH])
        next_words.append(tokens[i + SEQ_LENGTH])

    # Mini-batch stocastic
    print('Training...')
    for X, y, epoch in dp.create_minibatch(text, BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS):
        rnn.train_on_batch(X, y)

    # Save trained model to pickle file
    filename = '../model/rnn.pkl'
    with open(filename, 'w') as f:
        pickle.dump(rnn, f)
        print('Your model has been pickled')
    return None

if __name__ == '__main__':
    with open('../../soif_data/text/book1.txt') as f:
        text = f.read()
        text = ''.join(char for char in text.lower() if char not in punctuation)
        text = text.split(' ')
    train_rnn(text)
