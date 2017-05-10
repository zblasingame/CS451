"""Script to test classification accuracy of iris dataset

Author: Zander Blasingame
Class: CS 451
"""

import numpy as np
import tensorflow as tf
import csv
import random

from classifier import Classifier

net = Classifier(4, 750, 3, batch_size=10,
                 num_epochs=1000, display=True)

net.train('data.csv')
acc = net.test('data.csv')
print(acc)
