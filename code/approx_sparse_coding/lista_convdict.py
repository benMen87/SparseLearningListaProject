import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC


class LISTAConvDict (ApproxSC):
    """description of class"""

    def __init__(self, We_shape, unroll_count, We=None,
                 shrinkge_type='soft thresh', batch_size=1,
                 shared_threshold=False, kernal_size=3, amount_of_kernals=16):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super(LISTAConv, self).__init__(We_shape, unroll_count,
                                        shrinkge_type, shared_threshold, batch_size)

        self.input_channels = 1  # TODO: add support for RGB?
        self.kernal_size = kernal_size
        self.amount_of_kernals = amount_of_kernals

        S_shape = [self.kernal_size, self.kernal_size, self.input_channels, self.amount_of_kernals]
        self._S = tf.Variable(tf.truncated_normal(S_shape), name='S')
        if We is not None:
            #
            # warm start
            L = max(abs(np.linalg.eigvals(np.matmul(We, We.T))))
            self._theta = [tf.Variable(tf.constant(0.5/L, shape=[1, self.output_size], dtype=tf.float32), name='theta') for _ in range(unroll_count)]
            self._We = tf.Variable(We.T/L, name='We', dtype=tf.float32)
        else:
            self._theta = [tf.Variable(tf.truncated_normal([1, self.output_size]),
                                       name='theta') for _ in range(unroll_count)]

            self._We = tf.Variable(tf.truncated_normal([self.input_size,
                                                        self.output_size]),
                                   name='We', dtype=tf.float32)

    def _lista_step(self, Z, B, S, theta, shrink_fn):
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
        shape = Z.get_shape().as_list()
        im_size = tf.to_int32(tf.sqrt(tf.to_float(shape[1])))
        Z_reshpe = tf.reshape(Z, [-1,
                                  im_size,
                                  im_size,
                                  self.input_channels])

        ZcovS = tf.nn.conv2d(Z_reshpe, S, [1, 1, 1, 1], padding='SAME')
        ZcovS_maxpool = tf.reduce_mean(ZcovS, reduction_indices=[3], keep_dims=True)
        ZcovS_flat = tf.reshape(ZcovS_maxpool, (-1, shape[1]))
        C = B + ZcovS_flat
        Z = shrink_fn(C, theta)
        return (Z, B)

    def build_model(self):

        B = tf.matmul(self._X, self._We)
        shrinkge_fn = self._shrinkge()
        self._Z = shrinkge_fn(B, self._theta[0])
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            self._Z, B = self._lista_step(self._Z, B, self._S,
                                          self._theta[t], shrinkge_fn)

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_sum(tf.reduce_mean((self.output -
                                                           self.target) ** 2, 0))
                self._loss /= 2
        return self._loss

    @property
    def output(self):
        return self._Z

    @property
    def batch_size(self):
        return self.train_batch_size
