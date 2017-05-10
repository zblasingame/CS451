"""Perceptron game

Author: Zander Blasingame
Class: CS 451
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Network paramaters
learning_rate = 0.1
reg_param = 0.01
num_input = 2

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
saver = tf.train.Saver({'weights': weights, 'biases': biases})

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, './model.ckpt')

# for gpu
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


def train(training_data, test=False):
    """Trains perceptron

    Args:
        training_data (list of dicts): Training data formated as [{'X', 'Y'}]
    """

    np.random.shuffle(training_data)

    trX = [entry['X'] for entry in training_data]
    trY = [entry['Y'] for entry in training_data]

    feed_dict = {X: np.atleast_2d(trX),
                 Y: np.atleast_1d(trY)}

    with tf.Session() as sess:
        saver.restore(sess, './model.ckpt')
        for i in range(25):
            sess.run(optimizer, feed_dict=feed_dict)
        print(sess.run(cost, feed_dict=feed_dict))

        saver.save(sess, './model.ckpt')

        if test:
            pass


def get_desicion_boundary():
    """Finds desicion boundary of perceptron

    Returns:
        (tuple):  Returns tuple of (m,b) for the form y=mx+b
    """
    with tf.Session() as sess:
        saver.restore(sess, './model.ckpt')
        w = sess.run(weights)
        b = sess.run(biases)

        print(w)
        print(b)

        return -w[0]/w[1], -b/w[1]


def get_data_point():
    x = float(input('Enter x coordinate (float): '))
    y = float(input('Enter y coordinate (float): '))
    c = int(input('Enter class label (0 or 1): '))

    return {'x': x, 'y': y, 'class': c}


# Prompt for initial dimenions
plt.axis([-10, 10, -10, 10])
plt.ion()
plt.show()
axes = plt.gca()
desicion_boundary = plt.plot([-10, 10], [-10, 10], 'b')

print('Classes are as follows: 0 = red, 1 = green')

_cluster = input('Generate two cluster test data? (y/n): ')
cluster = _cluster == 'y'

# generate initial test data
if cluster:
    c0 = (np.random.randn()*4, np.random.randn()*4)
    c1 = (np.random.randn()*4, np.random.randn()*4)

    data = []

    for i in range(25):
        x = c0[0] + np.random.randn()
        y = c0[1] + np.random.randn()
        plt.scatter(x, y, c='r')

        data.append({
            'X': [x, y],
            'Y': 0
        })

    for i in range(25):
        x = c1[0] + np.random.randn()
        y = c1[1] + np.random.randn()
        plt.scatter(x, y, c='g')

        data.append({
            'X': [x, y],
            'Y': 1
        })

    # update and draw
    plt.pause(0.01)

    train(data, True)

    m, b = get_desicion_boundary()
    f = lambda x: m*x + b

    # update desicion boundary
    axes.lines.remove(axes.lines[0])
    desicion_boundary = plt.plot([-10, 10], [f(-10), f(10)], c='b')


_batch = input('(Mini) Batch training? (y/n): ')
batch = _batch == 'y'

if batch:
    num_points = int(input(('Enter the number of data points '
                            'you wish to enter before training: ')))

    training_data = []

# Start game
while True:
    # Prompt user for input
    point = get_data_point()

    # Plot point
    color = 'r' if point['class'] == 0 else 'g'
    plt.scatter(point['x'], point['y'], c=color)
    plt.pause(0.001)

    training_point = {
        'X': [point['x'], point['y']],
        'Y': point['class']
    }

    if batch:
        training_data.append(training_point)
        num_points -= 1

        if num_points == 0:
            print('Training')
            train(training_data)

            training_data = []
            print('Training buffer emptied')

            num_points = int(input(('Enter the number of data points '
                                    'you wish to enter before training: ')))
    else:
        train([training_point], True)

    m, b = get_desicion_boundary()
    print('m={}'.format(m))
    print('b={}'.format(b))

    f = lambda x: m*x + b

    # update desicion boundary
    axes.lines.remove(axes.lines[0])
    desicion_boundary = plt.plot([-10, 10], [f(-10), f(10)], c='b')
