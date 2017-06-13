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

def train_rnn(raw_data):
    '''
    Trains a word-level recurrent neural network using LSTM architecture on text
    corpora.

    Parameter
    ---------
    raw_data: (iterable) text corpora
    '''
    text, vocab, unknown_tokens = dp.text_to_vocab(raw_data, vocab_size=VOCAB_SIZE)
    data = dp.text_to_sequence(text, vocab)

    # Build a model: see hyper parameters above
    print('Building model...')
    rnn = Sequential([
        Embedding(VOCAB_SIZE, INTERNAL_SIZE, input_length=SEQ_LENGTH),
        LSTM(INTERNAL_SIZE, return_sequences=True, input_shape=(None, SEQ_LENGTH, INTERNAL_SIZE)),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE),
        Dropout(DROPOUT),
        Dense(INTERNAL_SIZE),
        Activation('softmax')])

    optimizer = Adam(lr=LEARNING_RATE)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')

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
