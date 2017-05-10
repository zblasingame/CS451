"""Script to generate text

Author:     Zander Blasingame
Class:      CS 451
"""

import tensorflow as tf
import numpy as np
import json

import models.Basic_GAN

# Network parameters
learning_rate   = 0.01
dropout         = 0.5
latent_vector   = 100
real_vector     = 1000

# Create GAN
G = {
    'sizes': [latent_vector, 500, real_vector],
    'activations': [tf.nn.relu, tf.nn.relu]
}

D = {
    'sizes': [real_vector, 200, 1],
    'activations': [tf.nn.relu, tf.identity]
}

gan = models.Basic_GAN.Basic_GAN(G, D)

X = tf.placeholder(tf.float32, shape=[None, real_vector], name='X')
Z = tf.placeholder(tf.float32, shape=[None, latent_vector], name='Z')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

G_sample        = gan.generator(Z, dropout)
D_logit_real    = gan.discriminator(X, dropout)
D_logit_fake    = gan.discriminator(G_sample, dropout)
D_real          = tf.nn.sigmoid(D_logit_real)
D_fake          = tf.nn.sigmoid(D_logit_fake)

# Losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_real, labels=tf.ones_like(D_logit_real)
))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)
))
D_loss = D_loss_fake + D_loss_real

# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)
# ))
G_loss = 0.5 * tf.reduce_mean(tf.subtract(tf.log(D_fake + 1e-8),
                                          tf.log(1 - D_fake + 1e-8))**2)

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
D_solver = D_solver.minimize(D_loss, var_list=gan.get_dis_weights())
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
G_solver = G_solver.minimize(G_loss, var_list=gan.get_gen_weights())

init_op = tf.global_variables_initializer()

# File parsing ops
language_data = {}
with open('data/c-out.json', 'r') as f:
    language_data = json.load(f)


def data_generator():
    i = 0
    text = []
    while True:
        with open('data/c-code/{}'.format(i), 'r') as f:
            text = f.read()

        text = text[0:real_vector]

        if i < 800:
            i += 1
        else:
            i = 0

        data = np.array([language_data['c2i'][c] for c in text])

        yield data


# Helper functions
def sample_Z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])


def _conv_text(index):
    if index >= language_data['vocab_size']:
        index = language_data['vocab_size'] - 1

    return language_data['i2c'][str(index)]

convert_to_text = np.vectorize(_conv_text)


with tf.Session() as sess:
    sess.run(init_op)

    get_next_batch = data_generator()

    for epoch in range(10000):
        X_mb = np.atleast_2d(next(get_next_batch))

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
            X: X_mb,
            Z: sample_Z(1, latent_vector),
            keep_prob: dropout
        })

        for i in range(1):
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                Z: sample_Z(1, latent_vector),
                keep_prob: dropout
            })

        if epoch % 1000 == 0:
            print('-'*79)
            print('Epoch: {}'.format(epoch))
            print('D_loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('Generated Text: ')

            num_text = sess.run(G_sample, feed_dict={
                Z: sample_Z(1, latent_vector),
                keep_prob: 1
            })

            str_text = convert_to_text(num_text.astype(int))

            print(''.join(str_text[0].tolist()))
