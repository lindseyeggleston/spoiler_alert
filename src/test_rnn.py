import h5py
import numpy as np
import data_processing as dp
from keras_rnn import _vectorize_text

BATCH_SIZE = 1
SEQ_LENGTH = 30
STEP = 1
OUTPUT_LEN = 50

def tokens_to_text(tokens, indices_word):
    '''
    Converts predicted token sequence into text.

    Parameters
    ----------
    tokens: ARRAY/LIST - iterable of predicted token indices
    indices_word: DICT - dictionary where keys are word indices and values are
        words in text vocabulary

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

def test_rnn(model, text, word_indices, batch_size=BATCH_SIZE):
    '''
    Compares the generated text prediction to known subsequent text

    Parameters
    ----------
    model: fitted/trained keras RNN model
    text: STR - text to predict on
    word_indices: DICT - vocab dictionary for training data where keys are words
        (str) and values are indices (int)
    batch_size: INT - batch size for updating weights

    Returns
    -------
    None
    '''
    indices_word = dict((v, k) for k, v in word_indices.items())

    tokens = dp.tokenize_text(text)
    X, y = _vectorize_text(tokens, word_indices, seq_length=SEQ_LENGTH, step=STEP)
    predict = model.predict(X, batch_size=batch_size)
    predict_, y_ = tokens_to_text(predict), tokens_to_text(y)

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

    predictions = []
    for i in range(output_len):
        X = tokens_[i: i+SEQ_LENGTH]
        Xo = np.zeros((1,SEQ_LENGTH, 5000))
        for j, val in enumerate(X):
            Xo[0,j,val] = 1
        predict = model.predict_on_batch(Xo)
        predictions.append(np.argmax(predict))
        tokens_.append(np.argmax(predict))

    text = tokens_to_text(predictions, indices_word) + "…"
    return text
