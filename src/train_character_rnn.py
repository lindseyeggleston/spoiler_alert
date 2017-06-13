from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

SEQ_LENGTH = 30
BATCH_SIZE = 300
INTERNAL_SIZE = 512 # number of nodes in each hidden layer
DROPOUT = 0.2
LEARNING_RATE = 0.001
ALPHABET_SIZE = 98 # number of acceptable characters, numbers, and punctuation


def train_rnn(text):

    # Construct model with 3 hidden LSTM layers
    print('Building model...')
    rnn = Sequential([
        Embedding(ALPHABET_SIZE, ALPHABET_SIZE)
        LSTM(INTERNAL_SIZE, return_sequences=True, input_shape=(None, SEQ_LENGTH, ALPHABET_SIZE)),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(INTERNAL_SIZE, return_sequences=False),
        Dropout(DROPOUT),
        Dense(ALPHABET_SIZE),
        Activation('softmax')])

    # Learning rate optimization (RMSprop or Adam)
    optimizer = Adam(lr=Learning)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')
