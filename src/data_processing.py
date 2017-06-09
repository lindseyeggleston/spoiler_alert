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

def create_minibatch(text, batch_size, seq_length, epochs):
    data = np.array(text.strip('\n').split(' '))
    num_batches = (data.size - 1) // (batch_size * seq_length)


if __name__ == '__main__':
    with open('../data/SOIF/AGameOfThrones.txt') as f:
        text = f.read()
        raw_data = text.split('\n')
    vocab = text_to_vocab(raw_data, 5000)
    print('Length: ', len(vocab))
