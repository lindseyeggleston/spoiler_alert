import numpy as np
import tensorflow as tf
import os
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

epochs = 3
sequence_len = 30 # can this vary in size?
batch_size = 200
alphabet_size = 98 # includes letters, numbers, and punctuation
chunk_size = 28
n_layers = 3
learning_rate = 0.001 # look into adaptive learning rates
dropout = 0.8

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(X):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

    X = tf.transpose(X, [1,0,2])
    X = tf.reshape(X, [-1, chunk_size])
    X = tf.split(0, n_chunks, X)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell, X, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(X):
    prediction = recurrent_neural_network(X)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks,\
                chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
