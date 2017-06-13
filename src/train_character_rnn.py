from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

seq_length = 30
batch_size = 300
internal_size = 512 # number of nodes in each hidden layer
dropout = 0.2
learning_rate = 0.001
alphabet_size = 98 # number of acceptable characters, numbers, and punctuation


def train_rnn(text):

    # Construct model with 3 hidden LSTM layers
    print('Building model...')
    rnn = Sequential([
        LSTM(internal_size, return_sequences=True, input_shape=(None, seq_length)),
        Dropout(dropout),
        LSTM(internal_size, return_sequences=True),
        Dropout(dropout),
        LSTM(internal_size, return_sequences=False),
        Dropout(dropout),
        Dense(vocab_size),
        Activation('softmax')])

    # Learning rate optimization (RMSprop or Adam)
    optimizer = Adam(lr=learning_rate)
    rnn.compile(optimizer=optimizer, loss='categorical_crossentropy')
