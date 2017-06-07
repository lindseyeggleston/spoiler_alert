import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers

# model parameters
epochs = 3 # add in more epochs later
sequence_len = 100 # can this vary in size?
batch_size = 200
alphabet_size = 98 # includes letters, numbers, and punctuation
internal_size = 512 # where does this size come from?
n_layers = 3
learning_rate = 0.001 # look into adaptive learning rates
dropout = 0.8

# the model
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [batch_size, sequence_len]
Xo = tf.one_hot(X, alphabet_size, 1.0, 0.0)             # [batch_size, sequence_len, alphabet_size]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y = tf.placeholder(tf.uint8, [None, None], name='Y_hat')  # [batch_size, sequence_len]
Yo = tf.one_hot(Y_, alphabet_size, 1.0, 0.0)               # [batch_size, sequence_len, alphabet_size]
# input state
Hin = tf.placeholder(tf.float32, [None, internal_size*n_layers], name='Hin')  # [batch_size, internal_size * n_layers]

def recurrent_neural_network(X):
    lstm_cells = [rnn.BasicLSTMCell(internal_size) for _ in range(n_layers)]
    dropout_cells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in lstm_cells]
    multicell = rnn.MultiRNNCell(dropout_cells, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

    Yin, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
    Yflat = tf.reshape(Yin, [-1, internal_size])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.fully_connected(Yflat, alphabet_size, activation_fn=None)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Yflat_ = tf.reshape(Yo, [-1, alphabet_size])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
    Yo_hat = tf.nn.softmax(Ylogits, name='Yo_hat')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y_hat = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y_hat = tf.reshape(Y, [batchsize, -1], name="Y_hat")  # [ BATCHSIZE, SEQLEN ]

    return Y_hat

def train_neural_network(X, n_epochs=epochs):
    prediction = recurrent_neural_network(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
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
