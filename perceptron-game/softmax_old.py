"""Script to test classification accuracy of iris dataset

Classify if iris is I. setosa or not

Author: Zander Blasingame
Class: CS 451
"""

import numpy as np
import tensorflow as tf
import csv
import random

from NeuralNet import NeuralNet

# Load data
data = []

with open('data.csv', 'r') as f:
    for row in csv.reader(f):
        row[-1] = 1 if row[-1] == 'Iris-setosa' else 0
        data.append(row)

# split dataset
random.shuffle(data)
stop = int(0.7 * len(data))
train_data = data[:stop]
test_data = data[stop:]

trX = np.array([row[:-1] for row in train_data]).astype(np.float)
trY = np.array([row[-1] for row in train_data])
training_size = trX.shape[0]

teX = np.array([row[:-1] for row in test_data]).astype(np.float)
teY = np.array([row[-1] for row in test_data])
testing_size = trY.shape[0]

# Network Parameters
eta = 0.001
dropout_prob = 0.5
reg_param = 0.01
training_epochs = 25
display_step = 1

num_input = trX.shape[1]
num_classes = 2

X = tf.placeholder('float', [None, num_input], name='X')
Y = tf.placeholder('int64', [None], name='Y')
keep_prob = tf.placeholder('float')

# Create classifier
net = NeuralNet([num_input, 25, num_classes], [tf.nn.relu, tf.identity])

logits = net.create_network(X, keep_prob)
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=Y)
cost += reg_param + net.get_l2_loss()

adam = tf.train.AdamOptimizer(learning_rate=eta)
train_fn = adam.minimize(cost)

# for evaluations
preds = tf.equal(tf.argmax(logits, 1), Y)
accuracy = tf.reduce_mean(tf.cast(preds, 'float'))

init_op = tf.global_variables_initializer()

# Train and eval
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(training_size):
            feed_dict = {
                X: np.atleast_2d(trX[i]),
                Y: np.atleast_1d(trY[i]),
                keep_prob: dropout_prob
            }

            _, c = sess.run([train_fn, cost], feed_dict=feed_dict)

            avg_cost += c[0] / training_size

        if epoch % display_step == 0:
            print(('Epoch: {0:03} with '
                   'cost={1:.5f}').format(epoch+1, avg_cost))

    print('Optimization Finished')

    print(sess.run(preds, feed_dict={X: teX, Y: teY, keep_prob: 1}))

    print('Accuracy: {}%'.format(sess.run(accuracy, feed_dict={
        X: teX,
        Y: teY,
        keep_prob: 1
    }) * 100))
