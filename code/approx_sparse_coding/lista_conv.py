import os
import sys
import tensorflow as tf
import numpy as np


class LISTAConv():
    """description of class"""

    def __init__(self, We_shape, unroll_count, We=None,
                 shrinkge_type='soft thresh', batch_size=1,
                 kernal_size=3, amount_of_kernals=16):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super(LISTAConv, self).__init__(We_shape, unroll_count,
                                        shrinkge_type, batch_size)

        self.input_channels = 1  # TODO: add support for RGB?
        self.kernal_size = kernal_size
        self.amount_of_kernals = amount_of_kernals
        self.input_size = input_size
        self.target_size = target_size

        S_shape = [self.kernal_size, self.kernal_size, self.input_channels, self.amount_of_kernals]
        if We is not None:
            #
            # warm start
            L = max(abs(np.linalg.eigvals(np.matmul(We, We.T))))
            self._theta = tf.Variable(tf.constant(0.5/L, shape=[1, self.output_size],
                                      dtype=tf.float32), name='theta')
            self._We = tf.Variable(We.T/L, name='We', dtype=tf.float32)
        else:
            self._theta = tf.Variable(tf.truncated_normal([1, self.output_size]),
                                      name='theta')
            self._We = tf.Variable(tf.truncated_normal([self.input_size,
                                                        self.output_size]),
                                   name='We', dtype=tf.float32)

    def _lista_step(self, Z, B, S, shrink_fn):
        """ LISTA step.

        Args:
            Z:    sparse representation of last iteration.
            B:    result of B after last iteration.
            shrink_fn: type of shrinkage function to use. 
            S:         is not updated every unrolling rather via GD. 
        Returns:
            update Z and B
        """
        #
        # run one lcod pass through
        ZcovS = tf.nn.conv2d(Z, S, [1, 1, 1, 1], padding='SAME')
        ZcovS_maxpool = tf.nn.max_pool(ZcovS, [1, 1, self.amount_of_kernals],
                                       [1, 1, self.amount_of_kernals])
        ZcovS_flat = tf.reshape(ZcovS_maxpool, target_size)
        C = B + ZcovS_flat
        Z = shrink_fn(C)
        return (Z, B)

    def build_model(self):

        B = tf.matmul(self._X, self._We)
        shrinkge_fn = self._shrinkge()
        self._Z = shrinkge_fn(B)
        #
        # run unrolling
        for t in range(self._unroll_count):
            self._Z, B = self._lista_step(self._Z, B, self._S, shrinkge_fn)

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_sum(tf.reduce_mean((self._output -
                                                           self._Z) ** 2, 0))
                self._loss /= 2
        return self._loss

    @property
    def output(self):
        return self._Z
