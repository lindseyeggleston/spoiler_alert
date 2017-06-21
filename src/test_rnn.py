import h5py
import numpy as np
import data_processing as dp
from keras_rnn import _vectorize_text
from nltk.tokenize import word_tokenize
from keras.models import load_model
from sys import argv

BATCH_SIZE = 1
SEQ_LENGTH = 30
STEP = 1
OUTPUT_LEN = 30

def tokens_to_text(tokens, indices_word, precedes_unknown_token):
    '''
    Converts predicted token sequence into text.

    Parameters
    ----------
    tokens: ARRAY/LIST - iterable of predicted token indices
    indices_word: DICT - dictionary where keys are word indices and values are
        words in text vocabulary
    precedes_unknown_token: DICT - dictionary of unknown tokens where keys are
        preceding words and values are lists of subsequent unknown tokens

    Returns
    -------
    a string of predicted text
    '''
    vec = np.vectorize(lambda x: indices_word[x])
    tokens_ = vec(np.array(tokens))
    text = ' '.join(tokens_)
    text = text.replace(' END_QUOTE', '”')
    text = text.replace('START_QUOTE ', '“')
    text = text.replace(' .', '.')

    return text

def load_model_and_dicts(model_path, vocab_path, unknown_path):
    '''
    Loads trained model from .h5 file and unpickles word_indices dictionary and
    precedes_unknown_token dictionary.

    Parameters
    ----------
    model_path: STR - filepath to saved trained model
    vocab_path: STR - filepath to word_indices pickle
    unknown_path: STR - filepath to precedes_unknown_token pickle

    Returns
    -------
    trained rnn model, word_indices dict, and precedes_unknown_token dict
    '''
    rnn = load_model(model_path)
    with open(vocab_path, 'rb') as f:
        word_indices = pickle.load(f)
    with open(unknown_path, 'rb') as f:
        unknown_path = pickle.load(f)
    return rnn, word_indices, precedes_unknown_token

def test_rnn(model_path, text, word_indices, batch_size=BATCH_SIZE):
    '''
    Compares the generated text prediction to known subsequent text

    Parameters
    ----------
    model_path: filepath to fitted/trained keras RNN model saved as .h5
    text: STR - text to predict on
    word_indices: DICT - vocab dictionary for training data where keys are words
        (str) and values are indices (int)
    batch_size: INT - batch size for updating weights

    Returns
    -------
    None
    '''
    rnn = load_model(model_path)
    indices_word = dict((v, k) for k, v in word_indices.items())

    tokens = dp.tokenize_text(text)
    X, y = _vectorize_text(tokens, word_indices, seq_length=SEQ_LENGTH, step=STEP)
    predict = model.predict(X, batch_size=batch_size)
    predict_ = tokens_to_text(predict, indices_word, precedes_unknown_token)
    y_ = tokens_to_text(y, indices_word, precedes_unknown_token)

    print('\nActual text')
    print(y_)

    print('\nNew text')
    print(predict_)

def generate_one_prediction(model, text, word_indices, output_len=OUTPUT_LEN, batch_size=BATCH_SIZE):
    '''
    Generates new text

    Parameters
    ----------
    model: fitted/trained keras RNN model
    text: STR - text to predict on
    word_indices: DICT - vocab dictionary for training data where keys are words
        (str) and values are indices (int)
    output_len: INT - length of desired output sequence
    batch_size: INT - batch size for updating weights

    Returns
    -------
    a string of new text
    '''
    indices_word = dict((v, k) for k, v in word_indices.items())
    tokens = dp.tokenize_text(text)
    vec = np.vectorize(lambda x: word_indices[x])
    tokens_ = list(vec(np.array(tokens)))

    predictions = np.zeros((5,30))
    for i in range(output_len):
        X = tokens_[i: i+SEQ_LENGTH]
        Xo = np.zeros((1,SEQ_LENGTH, 5000))
        for j, val in enumerate(X):
            Xo[0,j,val] = 1
        predict = model.predict_on_batch(Xo)
        top_5 = np.argsort(predict)[-5::-1]
        predictions[i] = top_5
        r = np.random.randint(5)
        tokens_.append(predictions[i,r])

    vec2 = np.vectorize(lambda x: indices_word[x])
    predict_ = vec2(predictions)
    return predict_

if __name__ == '__main__':
    _, model_path, vocab_path, unknown_path = argv
    rnn, word_indices, precedes_unknown_token = load_model_and_dicts(model_path,\
            vocab_path, unknown_path)
