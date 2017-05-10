"""Trains and tests MLP classifier

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import numpy as np
import tensorflow as tf
import pandas as pd

from MLP import MLP


class Classifier:
    """MLP classifier for detecting OpenSSL version
    Has training and testing methods
    """

    def __init__(self, num_input, num_units, num_classes,
                 batch_size=100, num_epochs=10, display=False,
                 blacklist=[], whitelist=[]):
        """Creates classifier for finding the version"""

        # Network parameters
        self.l_rate = 0.001
        self.dropout_prob = 1
        self.training_epochs = num_epochs
        self.display_step = 10
        self.batch_size = batch_size
        self.display = display

        self.blacklist = blacklist
        self.whitelist = whitelist

        assert not (self.blacklist and self.whitelist), (
            'Both whitelist and blacklist are defined'
        )

        # Placeholders
        self.X = tf.placeholder('float', [None, num_input], name='X')
        self.Y = tf.placeholder('int64', [None], name='Y')
        self.keep_prob = tf.placeholder('float')

        # Create Network
        self.mlp = MLP([num_input, num_units, num_classes],
                       [tf.nn.relu, tf.identity])

        logits = self.mlp.create_network(self.X, self.keep_prob)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.Y)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
        self.optimizer = self.optimizer.minimize(self.cost)

        # for evaluation
        predictions = tf.equal(tf.argmax(logits, 1), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # for gpu
        self.config = tf.ConfigProto(log_device_placement=False)

    def train(self, train_file=''):
        """Trains classifier
        Training file must be csv formatted
        """

        trX, trY = grab_data(train_file, self.blacklist, self.whitelist)
        training_size = len(trX)

        assert self.batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            for epoch in range(self.training_epochs):
                avg_cost = 0
                for i in range(0, training_size, self.batch_size):
                    # for batch training
                    upper_bound = i + self.batch_size
                    if upper_bound >= training_size:
                        upper_bound = training_size - 1

                    feed_dict = {self.X: np.atleast_2d(trX[i:upper_bound]),
                                 self.Y: np.atleast_1d(trY[i:upper_bound]),
                                 self.keep_prob: self.dropout_prob}
                    _, c = sess.run([self.optimizer, self.cost],
                                    feed_dict=feed_dict)

                    avg_cost += c[0] / training_size

                if epoch % self.display_step == 0:
                    self.print(('Epoch: {0:03} with '
                                'cost={1:.9f}').format(epoch+1, avg_cost))

            self.print('Optimization Finished')

            # save model
            save_path = self.saver.save(sess, './model.ckpt')
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, test_file=''):
        """Trains classifier
        Training file must be csv formatted
        """

        teX, teY = grab_data(test_file, self.blacklist, self.whitelist)

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, './model.ckpt')

            _accuracy = sess.run(self.accuracy,
                                 feed_dict={self.X: teX,
                                            self.Y: teY,
                                            self.keep_prob: 1.0})
            _accuracy *= 100

            self.print('accuracy={}'.format(_accuracy))

            return _accuracy

    def print(self, val):
        if self.display:
            print(val)


def grab_data(filename, blacklist=[], whitelist=[]):
    data = pd.read_csv(filename)

    assert not (blacklist and whitelist), (
        'Both whitelist and blacklist are defined'
    )

    names = data.columns[1:]

    if not whitelist:
        for entry in blacklist:
            data = data.drop(entry, 1)
    else:
        for name in names:
            if name not in whitelist:
                data = data.drop(name, 1)

    X = data.values[:, :-1]
    Y = data.values[:, -1]

    # Parse labels into indicies [0...N]
    uniq = list(set(Y))  # list of unique entries

    _Y = [uniq.index(y) for y in Y]

    return X.astype(np.float), _Y


def get_dimensions(filename, blacklist=[], whitelist=[]):
    data = pd.read_csv(filename)

    assert not(blacklist and whitelist), (
        'Both whitelist and blacklist are defined'
    )

    names = data.columns[1:]

    if not whitelist:
        for entry in blacklist:
            data = data.drop(entry, 1)
    else:
        for name in names:
            if name not in whitelist:
                data = data.drop(name, 1)

    X = data.values[:, 1:]
    Y = data.values[:, 0]

    # Parse labels into indicies [0...N]
    uniq = list(set(Y))  # list of unique entries

    return len(X[0]), len(uniq)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help='Location of training file')
    parser.add_argument('--test_file',
                        type=str,
                        default=None,
                        help='Location of testing file')
    parser.add_argument('--num_units',
                        type=int,
                        default=10,
                        help='Number of units in the hidden layer')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Size of batch for training (mini SGD)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epochs')
    parser.add_argument('--normalize',
                        action='store_true',
                        help='Flag to normalize input data')

    args = parser.parse_args()

    filename = args.train_file if args.train_file else args.test_file

    num_input, num_classes = get_dimensions(filename)

    classifier = Classifier(num_input, args.num_units,
                            num_classes, args.batch_size,
                            args.epochs, True)

    if args.train_file:
        classifier.train(args.train_file)

    if args.test_file:
        classifier.test(args.test_file)
