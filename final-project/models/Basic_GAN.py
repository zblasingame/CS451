"""Implementation of a Basic GAN (Ian Goodfellow et al.)

Author:     Zander Blasingame
Class:      CS 451
"""

import tensorflow as tf

from models.NeuralNet import NeuralNet


class Basic_GAN:
    """Generative Adversial Network with simple Neural Nets.

    Args:
        generator (dict):
            Dictionary containing 'sizes' and 'activations'
        discriminator (dict):
            Dictionary containing 'sizes' and 'activations'
    """

    def __init__(self, generator, discriminator):
        """Init GAN"""
        with tf.variable_scope('generator'):
            self._generator = NeuralNet(generator['sizes'],
                                        generator['activations'])

        with tf.variable_scope('discriminator'):
            self._discriminator = NeuralNet(discriminator['sizes'],
                                            discriminator['activations'])

    def generator(self, Z, keep_prob):
        """Create the Generator network

        Args:
            Z (tf.Tensor):
                Placeholder Tensor with dimensions of the latent vector.
            keep_prob (tf.Tensor):
                Placholder Tensor of rank one containing the probability
                for the dropout technique.
        Returns:
            (tf.Tensor):
                A tensor to be evaluated containing the predicted
                of the generator.
        """
        return self._generator.create_network(Z, keep_prob)

    def discriminator(self, X, keep_prob):
        """Create the Generator network

        Args:
            X (tf.Tensor):
                Placeholder Tensor with dimensions of the real data.
            keep_prob (tf.Tensor):
                Placholder Tensor of rank one containing the probability
                for the dropout technique.

        Returns:
            (tf.Tensor):
                A tensor to be evaluated containing the predicted
                of the discriminator.
        """
        return self._discriminator.create_network(X, keep_prob)

    def get_gen_weights(self):
        """Get the generator weights.

        Returns:
            (list of tf.Variable):
                List of weights.
        """
        return self._generator.get_weights()

    def get_dis_weights(self):
        """Get the discriminator weights.

        Returns:
            (list of tf.Variable):
                List of weights.
        """
        return self._discriminator.get_weights()
