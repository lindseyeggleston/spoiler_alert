import numpy as np
import os
from keras.preprocessing.text import text_to_word_sequence, one_hot
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict, defaultdict, namedtuple

def file_to_text(filepath):
    if os.path.isfile(filepath):
        text = filepath.read().lower()
    elif os.path.isdir(filepath):
        with open(filepath + '/*.txt') as f:
            text = t.read().lower()
    return text

def text_to_vocab(text, vocab_length=None):
    '''
    Learns the vocabulary within a text and refines the length to the most
    frequently used words

    Parameters:
    -----------
    text: LIST/ARRAY - iterable which yields a string of text
    vocab_length: INT - num of words in vocab

    Returns:
    --------
    a vocab dictionary of specified length if vocab_length not None
    '''
    vectorizer = TfidfVectorizer(max_features=vocab_length)
    X = vectorizer.fit_transform(text)
    vocab = vectorizer.vocabulary_
    return vocab

def text_to_sequence(text):
    pass

def create_minibatch(text, batch_size, seq_length, num_epochs):
    '''
    Generates mini-batches of data to be processed within the recurrent neural network

    Parameters:
    -----------
    text: STR - text corpus
    batch_size: INT - size of batch
    seq_length: INT - length of sequence
    num_epochs: INT - num of epochs for training rnn

    Returns:
    --------
    yields input matrix X for a single batch, expected output y, and the current epoch
    '''
    data = np.array(text.split(' '))
    num_batches = data.size // (batch_size * seq_length)

    # Round and reshape data to be even with batch numbers
    rounded_data = num_batches * batch_size * seq_length
    x_data = np.reshape(data[0:rounded_data], [batch_size, num_batches * seq_length])
    y_data = np.reshape(data[1:rounded_data + 1], [batch_size, num_batches * seq_length])

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            X = x_data[:, seq_length * batch:seq_length * (batch + 1)]
            y = y_data[:, seq_length * batch:seq_length * (batch + 1)]

            # Shift by row(s) at each epoch for different looking data
            X = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)

            yield X, y, epoch  # Generator

if __name__ == '__main__':
    with open('../data/SOIF/AGameOfThrones.txt') as f:
        text = f.read()
        raw_data = text.split('\n')
    vocab = text_to_vocab(raw_data, 5000)
    print('Length: ', len(vocab))
