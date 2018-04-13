from keras.models import Model
from keras.layers import Input, Permute, Reshape, Dense, Lambda, RepeatVector
from keras.layers import GRU, LSTM, Bidirectional, Dropout, GlobalMaxPool1D
from keras.layers import BatchNormalization, multiply, Embedding
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


class BiLM():
    '''Bidirectional Language Model'''

    def __init__(self, n_neurons, max_seq_len, embed_size, vocab_size,
                 dropout, alpha):
        self.n = 0
        self.single_attention = False
        self.n_neurons = n_neurons
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.build()
        self.compile(alpha=alpha)

    def build(self):
        inp = Input(shape=(self.max_seq_len,))
        x = Embedding(self.vocab_size, self.embed_size,
                      weights=[embedding_matrix], trainable = True)(inp)
        x = self.recurrent_block(x)
        x = self.recurrent_block(x)
        x = self.recurrent_block(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(self.embed_size)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(1, activation='sigmoid')(x)  # initialize output weights
        self.model = Model(inputs=inp, outputs=x)

    def attention_3d_block(self, x, seq_len, single, n):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(x.shape[2])
        a = Permute((2, 1))(x)
        a = Reshape((input_dim, seq_len))(a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(seq_len, activation='softmax')(a)
        if single:
            a = Lambda(lambda x: K.mean(x, axis=1),
                       name=f'dim_reduction_{n}')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name=f'attention_vec_{n}')(a)
        output_attention_mul = multiply([x, a_probs], name=f'attention_mul_{n}')
        return output_attention_mul

    def recurrent_block(self, x, return_sequences=True):
        x = self.attention_3d_block(x, self.max_seq_len,
                               self.single_attention, self.n)
        self.n += 1  # this is needed so that each layer has a unique name
        x = Bidirectional(GRU(self.n_neurons, dropout=self.dropout,  # test GRU vs LSTM
                              return_sequences=return_sequences))(x)
        return BatchNormalization(axis=-1)(x)

    def compile(self, alpha):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(lr=alpha), metrics=['accuracy'])

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train, **kwargs)

    def predict(self):
        return self.model.predict()

    def summary(self):
        return self.model.summary()
