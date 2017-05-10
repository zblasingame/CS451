"""Script to generate images

Author:     Zander Blasingame
Class:      CS 451
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.slim as slim
import scipy.ndimage
import json

import models.Basic_GAN

# Network parameters
learning_rate   = 0.001
dropout         = 0.5
latent_vector   = 100
real_vector     = 65536  # 256x256

# weights initializer
initializer = tf.truncated_normal_initializer(stddev=0.2)


# Helper functions
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)


# Create GAN
def generator(Z):
    z_fc = slim.fully_connected(
        Z, 4*4*16*256, normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='generator',
        weights_initializer=initializer
    )
    z_con = tf.reshape(z_fc, [-1, 4, 4, 256])

    gen1 = slim.convolution2d_transpose(
        z_con, num_outputs=64, kernel_size=[5, 5], stride=[2, 2],
        padding='SAME', normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv1',
        weights_initializer=initializer
    )

    gen2 = slim.convolution2d_transpose(
        gen1, num_outputs=32, kernel_size=[5, 5], stride=[2, 2],
        padding='SAME', normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv2',
        weights_initializer=initializer
    )

    gen3 = slim.convolution2d_transpose(
        gen2, num_outputs=16, kernel_size=[5, 5], stride=[2, 2],
        padding='SAME', normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv3',
        weights_initializer=initializer
    )

    return slim.convolution2d_transpose(
        gen3, num_outputs=1, kernel_size=[32, 32], stride=[2, 2],
        padding='SAME', normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.tanh, scope='g_conv_out',
        weights_initializer=initializer
    )


def discriminator(X, reuse=False):
    dis1 = slim.convolution2d(
        X, 16, [4, 4], stride=[2, 2], padding='SAME',
        biases_initializer=None, activation_fn=lrelu,
        reuse=reuse, scope='d_conv1',
        weights_initializer=initializer
    )

    dis2 = slim.convolution2d(
        dis1, 32, [4, 4], stride=[2, 2], padding='SAME',
        normalizer_fn=slim.batch_norm, activation_fn=lrelu,
        reuse=reuse, scope='d_conv2',
        weights_initializer=initializer
    )

    dis3 = slim.convolution2d(
        dis2, 64, [4, 4], stride=[2, 2], padding='SAME',
        normalizer_fn=slim.batch_norm, activation_fn=lrelu,
        reuse=reuse, scope='d_conv3',
        weights_initializer=initializer
    )

    return slim.fully_connected(
        slim.flatten(dis3), 1, activation_fn=tf.nn.sigmoid,
        reuse=reuse, scope='d_out', weights_initializer=initializer
    )

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='X')
Z = tf.placeholder(tf.float32, shape=[None, latent_vector], name='Z')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

G_sample        = generator(Z)
D_real          = discriminator(X)
D_fake          = discriminator(G_sample, reuse=True)

# Losses
D_loss = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake))
G_loss = 0.5 * tf.reduce_mean(tf.square(
    tf.log(D_fake + 1e-8) - tf.log(1 - D_fake + 1e-8)
))

tvars = tf.trainable_variables()

print(tvars)

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
# D_solver = D_solver.minimize(D_loss, var_list=gan.get_dis_weights())
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
# G_solver = G_solver.minimize(G_loss, var_list=gan.get_gen_weights())

init_op = tf.global_variables_initializer()


def data_generator(batch_size=1):
    i = 0
    batch_count = 0
    batch = []
    while True:
        while batch_count < batch_size:
            data = scipy.ndimage.imread(
                'data/images/paintings/{}.jpg'.format(i),
                True
            )

            batch.append(data.flatten())

            batch_count += 1

            if i < 7:
                i += 1
            else:
                i = 0

        batch_count = 0

        yield np.array(batch)


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
        plt.imshow(sample.reshape(256, 256), cmap='Greys_r')

    return fig


with tf.Session() as sess:
    sess.run(init_op)

    get_next_batch = data_generator(4)

    for epoch in range(2001):
        X_mb = np.atleast_2d(next(get_next_batch))

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
            X: X_mb,
            Z: sample_Z(4, latent_vector),
            keep_prob: dropout
        })

        for i in range(1):
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                Z: sample_Z(4, latent_vector),
                keep_prob: dropout
            })

        if epoch % 100 == 0:
            print('-'*79)
            print('Epoch: {}'.format(epoch))
            print('D_loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))

            image_vecs = sess.run(G_sample, feed_dict={
                Z: sample_Z(4, latent_vector),
                keep_prob: 1
            })

            fig = plot(image_vecs)
            plt.savefig('output/{}.png'.format(epoch), bbbox_inches='tight')
            plt.close(fig)

    # Plot vector space
    print('-'*79)
    print('Plot of latent vector space')

    image_vecs = sess.run(G_sample, feed_dict={
        Z: np.linspace(-1, 1, 100*100).reshape(100, 100),
        keep_prob: 1
    })

    fig = plot(image_vecs, shape=(10, 10))
    plt.savefig('output/latent_vector.png'.format(epoch), bbbox_inches='tight')
    plt.close(fig)
