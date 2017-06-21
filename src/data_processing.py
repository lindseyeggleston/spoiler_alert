import numpy as np
import os
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
import glob

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
        with open(filepath) as f:
            text = f.read()
    elif os.path.isdir(filepath):
        text = ''
        for filename in glob.glob(filepath + '/*.txt'):
            with open(filename) as f:
                text += f.read()
    return text

def tokenize_text(text):
    '''
    Cleans text by seperating word contractions and punctuation.

    Parameter
    ---------
    text: STR - text corpus/corpora

    Returns
    -------
    list of words/tokens
    '''
    text = re.sub("’s", " ’s", text)
    text = re.sub("’m", " am", text)
    text = re.sub("’ve", " have", text)
    text = re.sub("’re", " are", text)
    text = re.sub("’d", " ’d", text)
    text = re.sub("’ll", " will", text)
    text = re.sub("n’t", " not", text)
    text = re.sub("’em", " them", text)
    text = text.replace('”', ' END_QUOTE')
    text = text.replace('“', 'START_QUOTE ')
    text = re.sub("…", "", text)

    # Tokenize text
    tokens = word_tokenize(text)
    return tokens

def text_to_vocab(tokens, vocab_size=8000):
    '''
    Learns the vocabulary within a text and refines the length to the most
    frequently used words while setting all else to UNKNOWN_TOKEN.

    Parameters:
    -----------
    tokens: ARRAY/LIST - text corpus/corpora
    vocab_size: INT - num of words in vocabulary

    Returns:
    --------
    a refined text (iterable), a word_indices (or vocab) dictionary of
    specified length, a dictionary of unknown_tokens and words that precede them
    '''
    # Find frequent words
    word_freq = Counter(tokens)
    assert (len(word_freq) >= vocab_size), \
            'There are {0} unique words in this text. Choose a smaller vocab size.'\
            .format(len(word_freq))
    if len(word_freq) == vocab_size:
        precedes_unknown_token = dict()
        word_indices = {word:i for i, word in enumerate(word_freq)}
    else:
        word_indices = {word[0]:i for i,word in enumerate(sorted(word_freq\
            .most_common(vocab_size-1)))}
        tokens, precedes_unknown_token = _create_unknown_token_dict(tokens, word_indices)

    return tokens, word_indices, precedes_unknown_token

def _create_unknown_token_dict(tokens, word_indices):
    '''
    Constructs a dictionary for referencing unknown tokens where keys are words
    that precede unknown tokens in a text and values are the subsequent unknown
    tokens then replaces unknown tokens with 'UNKNOWN_TOKEN'

    Parameters
    ----------
    tokens: LIST/ARRAY - iterable that contains tokens from a text corpus
    word_indices: DICT - vocabulary dictionary where keys are words and values
        are indices

    Returns
    -------
    a dictionary and refined token iterable
    '''
    precedes_unknown_token = defaultdict(list)
    for i, token in enumerate(tokens):
        if token not in word_indices:
            precedes_unknown_token[tokens[i-1]] += [token]
            tokens[i] = 'UNKNOWN_TOKEN'
    return tokens, precedes_unknown_token

def create_minibatch(text, batch_size, seq_length, n_epochs, word_indices):
    '''
    Generates mini-batches of data to be processed within the recurrent neural
    network

    Parameters
    ----------
    text: LIST - text corpus
    batch_size: INT - size of batch
    seq_length: INT - length of sequence
    n_epochs: INT - num of epochs for training rnn
    word_indices: DICT - refined vocab dictionary where keys are words and values
        are indices

    Returns
    -------
    yields input matrix X for a single batch, expected output y, and the current
    epoch
    '''
    n_batches = (len(text) - 1) // (batch_size * seq_length)

    # Round and reshape data to be even with batch numbers
    round_data = n_batches * batch_size * seq_length
    x_data = np.reshape(text[0:round_data], [batch_size, n_batches * seq_length])
    y_data = np.reshape(text[1:round_data + 1], [batch_size, n_batches * seq_length])

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X = x_data[:, seq_length * batch:seq_length * (batch + 1)]
            y = y_data[:, seq_length * batch:seq_length * (batch + 1)]

            X = np.roll(X, -epoch, axis=0)# Shift text for each epoch
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
