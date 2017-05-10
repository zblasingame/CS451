"""Script to generate images mnist

Author:     Zander Blasingame
Class:      CS 451
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.ndimage

import models.Conditional_GAN


mnist = input_data.read_data_sets('data/images/MNIST_data/', one_hot=True)

# Network parameters
learning_rate   = 0.001
dropout         = 0.5
latent_vector   = 50
X_dim           = mnist.train.images.shape[1]
Y_dim           = mnist.train.labels.shape[1]

# Create GAN
G = {
    'sizes': [latent_vector + Y_dim, 150, X_dim],
    'activations': [tf.nn.relu, tf.nn.sigmoid]
}

D = {
    'sizes': [X_dim + Y_dim, 150, 1],
    'activations': [tf.nn.relu, tf.identity]
}

gan = models.Conditional_GAN.Conditional_GAN(G, D)

X = tf.placeholder(tf.float32, shape=[None, X_dim], name='X')
Y = tf.placeholder(tf.float32, shape=[None, Y_dim], name='Y')
Z = tf.placeholder(tf.float32, shape=[None, latent_vector], name='Z')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

G_sample        = gan.generator(Z, Y, dropout)
D_logit_real    = gan.discriminator(X, Y, dropout)
D_logit_fake    = gan.discriminator(G_sample, Y, dropout)
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

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)
))
# G_loss = 0.5 * tf.reduce_mean(tf.square(
#     tf.log(D_fake + 1e-8) - tf.log(1 - D_fake + 1e-8)
# ))

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
D_solver = D_solver.minimize(D_loss, var_list=gan.get_dis_weights())
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
G_solver = G_solver.minimize(G_loss, var_list=gan.get_gen_weights())

init_op = tf.global_variables_initializer()


# Helper functions
def sample_Z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])


def plot(samples, shape=(2, 2)):
    fig = plt.figure(figsize=shape)
    gs = gridspec.GridSpec(shape[0], shape[1])
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


with tf.Session() as sess:
    sess.run(init_op)

    batch_size = 128
    n_sample = 16

    for epoch in range(50001):
        X_mb, Y_mb = mnist.train.next_batch(batch_size)

        for k in range(3):
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                X: X_mb,
                Y: Y_mb,
                Z: sample_Z(batch_size, latent_vector),
                keep_prob: dropout
            })

        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
            Z: sample_Z(batch_size, latent_vector),
            Y: Y_mb,
            keep_prob: dropout
        })

        if epoch % 1000 == 0:
            print('-'*79)
            print('Epoch: {}'.format(epoch))
            print('D_loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))

            image_vecs = sess.run(G_sample, feed_dict={
                Z: sample_Z(n_sample, latent_vector),
                Y: sample_Z(n_sample, Y_dim),
                keep_prob: 1
            })

            fig = plot(image_vecs, shape=(4, 4))
            plt.savefig('output/{}.png'.format(epoch), bbbox_inches='tight')
            plt.close(fig)

    # Plot vector space
    print('-'*79)
    print('Plot of latent vector space')

    y_sel_vec = [int(i/Y_dim) for i in range(100)]

    image_vecs = sess.run(G_sample, feed_dict={
        Z: sample_Z(100, latent_vector),
        Y: np.eye(Y_dim)[y_sel_vec],
        keep_prob: 1
    })

    fig = plot(image_vecs, shape=(10, 10))
    plt.savefig('output/latent_vector.png'.format(epoch), bbbox_inches='tight')
    plt.close(fig)
