import numpy as np
import os
from collections import Counter
from nltk.tokenize import word_tokenize

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
    text: STR - text corpus/corpora
    vocab_size: INT - num of words in vocab

    Returns:
    --------
    a refined text (iterable) with a vocab dictionary of specified length if vocab_size not None
    '''
    # Tokenize text
    tokens = word_tokenize(text)

    # Find frequent words
    word_freq = Counter(tokens)
    print(len(word_freq))
    if len(word_freq) < vocab_size:
        assert('The text contains {0} unique words. Select a smaller vocab size'\
            .format(len(word_freq)))
    elif len(word_freq) == vocab_size:
        n_freq_words = set([word[0] for word in word_freq.most_frequent(vocab_size)])
        unknown_tokens = None
    else:
        n_freq_words = {word[0]:i for i,word in enumerate(sorted(word_freq\
            .most_common(vocab_size-1)))}

        # convert all words not in refined vocab to 'UNKNOWN_TOKEN'
        n_freq_words['UNKNOWN_TOKEN'] = vocab_size - 1
        unknown_tokens = set([])
        for i, word in enumerate(tokens):
            if word not in n_freq_words.keys():
                unknown_tokens.add(word)
                tokens[i] = 'UNKNOWN_TOKEN'

    return tokens, n_freq_words, unknown_tokens

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
    with open('../../soif_data/characters/cersei.txt') as f:
        text = f.read()
    tokens, n_freq_words, unknown_tokens = text_to_vocab(text, vocab_size=7000)
    print(tokens[:500])
