"""Perceptron game

Author: Zander Blasingame
Class: CS 451
"""

import tensorflow as tf
import numpy as np
import csv

# Load data
_x = []
_y = []

with open('data.csv', 'r') as f:
    for row in csv.reader(f):
        _x.append(row[:-1])
        _y.append(1 if row[-1] == 'Iris-setosa' else 0)

dataX = np.array(_x).astype(np.float)
dataY = np.array(_y).astype(np.float)

# Network paramaters
learning_rate = 0.001
reg_param = 0.01
num_input = 4
batch_size = 10

X = tf.placeholder('float', [None, num_input], name='X')
Y = tf.placeholder('float', [None], name='Y')

# Create perceptron
weights = tf.Variable(tf.random_normal([num_input, 1]), name='weights')
biases = tf.Variable(tf.random_normal([1]), name='biases')

pred = tf.nn.sigmoid(tf.matmul(X, weights) + biases)
cost = tf.reduce_mean(tf.square(Y - pred))
cost += reg_param * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = optimizer.minimize(cost)

# init variables
init_op = tf.global_variables_initializer()

# Train and evaluate
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(1000):
        avg_cost = 0
        for i in range(0, len(dataX), batch_size):
            stop = i+batch_size if i+batch_size < len(dataX) else len(dataX)-1

            feed_dict = {
                X: np.atleast_2d(dataX[i:stop]),
                Y: np.atleast_1d(dataY[i:stop])
            }

            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)

            avg_cost += c / len(dataX)

        if epoch % 10 == 0:
            print('Epoch {:03} | Cost {:.5f}'.format(epoch+1, avg_cost))

    print('Optimization Finished')

    print('weights = {}'.format(sess.run(weights)))
    print('biases = {}'.format(sess.run(biases)))

    feed_dict = {
        X: dataX,
        Y: dataY
    }

    p = sess.run(pred, feed_dict=feed_dict)

    acc = 0

    for i in range(len(p)):
        acc += 1 if np.round(p[i]) == dataY[i] else 0
        # print('P = {} | Y = {}'.format(p[i], dataY[i]))

    acc /= len(p)

    print('Accuracy: {:.2f}%'.format(acc * 100))
