import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers
import pickle
import data_processing as dp

# model hyper parameters
N_EPOCHS = 10
SEQ_LENGTH = 30  # length of chunk fed into the RNN
BATCH_SIZE = 200
N_CHARS = 98  # number of characters; includes letters, numbers, and punct
INTERNAL_SIZE = 512  # size of hidden layer of neurons
N_LAYERS = 3
LEARNING_RATE = 0.001  # look into adaptive learning rates
DROPOUT = 0.8

# the model
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # percent keep/dropout param
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')
Xo = tf.one_hot(X, N_CHARS, 1.0, 0.0)

# expected outputs
Y = tf.placeholder(tf.uint8, [None, None], name='Y_hat')
Yo = tf.one_hot(Y, N_CHARS, 1.0, 0.0)

# hidden state
Hs = tf.placeholder(tf.float32, [None, INTERNAL_SIZE * N_LAYERS], name='Hs')


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
    lstm_cells = [rnn.BasicLSTMCell(INTERNAL_SIZE) for _ in range(N_LAYERS)]
    dropout_cells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell
                     in lstm_cells]
    multicell = rnn.MultiRNNCell(dropout_cells, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

    Y_, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32,
                              initial_state=Hs)
    return Y_, H


def train_neural_network(X, n_epochs=N_EPOCHS):
    Y_initial, H = recurrent_neural_network(X)

    Y_flat = tf.reshape(Y_initial, [-1, INTERNAL_SIZE])
    Y_logits = layers.fully_connected(Y_flat, N_CHARS, activation_fn=None)
    Y_flat_ = tf.reshape(Yo, [-1, N_CHARS])

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits,
                                                   labels=Y_flat_)
    cost = tf.reshape(loss, [batchsize, -1])
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    Yo_hat = tf.nn.softmax(Y_logits, name='Yo_hat')
    Y_hat = tf.argmax(Yo_hat, 1)
    Y_hat = tf.reshape(Y_hat, [batchsize, -1], name="Y_hat")

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                   labels=Yflat_)
    loss = tf.reshape(loss, [batchsize, -1])
    Yo = tf.nn.softmax(Ylogits, name='Yo')
    Y = tf.argmax(Yo, 1)
    Y = tf.reshape(Y, [batchsize, -1], name="Y")
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)),
                                      tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    print('Training...')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            for _ in range(int()):
                feed_dict = {x: epoch_x, y: epoch_y, Hin: istate,
                             lr: learning_rate, pkeep: dropout,
                             batchsize: batch_size}
                epoch_x, epoch_y = dp.create_minibatch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, seq_length))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x,
                                                              y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


with open('../../soif_data/SOIF/AGameOfThrones.txt') as f:
    text = f.read()
train_neural_network(text)
