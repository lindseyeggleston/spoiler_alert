import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers
import pickle
import data_processing as dp

# model hyper parameters
epochs = 10
seq_length = 30  # length of chunk fed into the RNN
batch_size = 200
chars = 98  # number of characters; includes letters, numbers, and punctuation
internal_size = 512  # size of hidden layer of neurons
n_layers = 3
learning_rate = 0.01  # look into adaptive learning rates
dropout = 0.8

# the model
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')  # shape = [batch_size, seq_length]
Xo = tf.one_hot(X, chars, 1.0, 0.0)  # [batch_size, seq_length, chars]

# expected outputs
Y = tf.placeholder(tf.uint8, [None, None], name='Y_hat')  # [batch_size, >= seq_length]
Yo = tf.one_hot(Y, chars, 1.0, 0.0)  # [batch_size, seq_length, chars]

# hidden state
Hs = tf.placeholder(tf.float32, [None, internal_size*n_layers], name='Hs')

def recurrent_neural_network(X):
    '''
    Constructs a character-level recurrent neural network with n_layers

    Parameters
    ----------
    X: sequence

    Returns
    -------
    A predicted sequence, Y_, and the hidden states
    '''
    print('Building a model...')
    lstm_cells = [rnn.BasicLSTMCell(internal_size) for _ in range(n_layers)]
    dropout_cells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell \
                        in lstm_cells]
    multicell = rnn.MultiRNNCell(dropout_cells, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

    Y_, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, \
                        initial_state=Hs)
    return Y_, H

def train_neural_network(X, n_epochs=epochs):
    Y_initial, H = recurrent_neural_network(X)

    Y_flat = tf.reshape(Y_initial, [-1, internal_size])  # [batch_size * seq_length, internal_size]
    Y_logits = layers.fully_connected(Y_flat, alphabet_size, activation_fn=None)   # [batch_size * seq_length, chars]
    Y_flat_ = tf.reshape(Yo, [-1, alphabet_size])     # [batch_size * seq_length, chars]

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_flat_)  # [batch_size * seq_length]
    cost = tf.reshape(loss, [batchsize, -1])      # [batch_size, seq_length]
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    Yo_hat = tf.nn.softmax(Y_logits, name='Yo_hat')        # [batch_size * seq_length, chars]
    Y_hat = tf.argmax(Yo_hat, 1)                          # [batch_size * seq_length]
    Y_hat = tf.reshape(Y_hat, [batchsize, -1], name="Y_hat")  # [batch_size, seq_length]

    print('Training...')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int()):
                feed_dict = {x: epoch_x, y: epoch_y, Hin: istate, lr: learning_rate, pkeep: dropout, batchsize: batch_size}
                epoch_x, epoch_y = dp.create_minibatch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, seq_length))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', n_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks,\
                chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
