import numpy as np
import os
from keras.preprocessing.text import text_to_word_sequence, one_hot
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict, defaultdict, namedtuple
from string import punctuation

def file_to_text(filepath):
    '''
    Opens text files and converts them into string objects

    Parameters
    ----------
    filepath: STR - path to file or folder

    Returns
    -------
    a string of text corpus/corpora
    '''
    if os.path.isfile(filepath):
        text = filepath.read().lower()
    elif os.path.isdir(filepath):
        with open(filepath + '/*.txt') as f:
            text = t.read()
    return text

def text_to_vocab(text, vocab_size=8000):
    '''
    Learns the vocabulary within a text and refines the length to the most
    frequently used words while setting all else to UNKNOWN_TOKEN.

    Parameters:
    -----------
    text: LIST/ARRAY - iterable which yields a string of text
    vocab_size: INT - num of words in vocab

    Returns:
    --------
    a refined text (iterable) with a vocab dictionary of specified length if vocab_size not None
    '''
    vectorizer = TfidfVectorizer(max_features=vocab_size-1)
    vectorizer.fit(text)
    vocab = vectorizer.vocabulary_
    if vocab_size != None:
        vocab['UNKNOWN_TOKEN'] = vocab_size - 1
    if len(vocab) < vocab_size:
        assert('The text contains {0} words. Please select a different vocab size'\
                    .format(len(vocab)))
    unknown_tokens = set([word for word in text if word not in vocab])
    new_text = ['UNKNOWN_TOKEN' if word not in vocab else word for word in text]
    return new_text, vocab, unknown_tokens

def word_to_indices(text, vocab):
    '''
    Converts array-like object from words into corresponding vocab index number

    Parameters
    ----------
    indices: ARRAY -
    vocab: DICT - text vocabulary where keys are words and values are index
            references
    Returns
    -------
    array-like object
    '''
    indices = [vocab[word] for word in text]
    return indices

def indices_to_word(indices, vocab):
    '''
    Converts array-like object from vocab index number into corresponding words

    Parameters
    ----------
    indices: ARRAY -
    vocab: DICT - text vocabulary where keys are words and values are index
            references
    Returns
    -------
    array-like object
    '''
    index_vocab = dict(list(zip(vocab.values(),vocab.keys())))
    return [index_vocab[index] for index in text]

def create_minibatch(text, batch_size, seq_length, num_epochs, vocab_size=8000):
    '''
    Generates mini-batches of data to be processed within the recurrent neural network

    Parameters:
    -----------
    text: LIST - text corpus
    batch_size: INT - size of batch
    seq_length: INT - length of sequence
    num_epochs: INT - num of epochs for training rnn

    Returns:
    --------
    yields input matrix X for a single batch, expected output y, and the current epoch
    '''
    num_batches = (len(text) - 1) // (batch_size * seq_length)

    # Round and reshape data to be even with batch numbers
    rounded_data = num_batches * batch_size * seq_length
    x_data = np.reshape(text[0:rounded_data], [batch_size, num_batches * seq_length])
    y_data = np.reshape(text[1:rounded_data + 1], [batch_size, num_batches * seq_length])

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            X = x_data[:, seq_length * batch:seq_length * (batch + 1)]
            y = y_data[:, seq_length * batch:seq_length * (batch + 1)]

            # Shift by row(s) at each epoch for different looking data
            X = np.roll(X, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)

            yield X, y, epoch  # Generator

def count_characters(text):
    '''
    Prints the number of different characters used in a text and creates a
    dictionary of character indexes

    Parameter
    ---------
    text: STR - text corpus

    Returns
    -------
    a dictionary with chacters as keys and index numbers as values
    '''
    char_set = set([])
    for char in text:
        char_set.add(char)
    print('This text contains {0} distinct characters'.format(len(char_set)))
    char_dict = {char:i for i,char in enumerate(sorted(list(char_set)))}
    return char_dict


if __name__ == '__main__':
    with open('../../soif_data/text/book1.txt') as f:
        text = f.read()
    #     text = ''.join(char for char in text if char not in punctuation)
    #     raw_data = text.lower().split(' ')
    # new_text, vocab, unknown_tokens = text_to_vocab(raw_data)
    # print('Length: ', len(vocab))
    # print(new_text[0:500])
    # print(unknown_tokens)
    # print(text_to_sequence(new_text, vocab))

    char_dict = count_characters(text)
    print(char_dict)
