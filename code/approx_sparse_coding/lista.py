import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC


class LISTA(ApproxSC):
    """description of class"""

    def __init__(self, We_shape, unroll_count, We=None,
                 shrinkge_type='soft thresh', shared_threshold=False, batch_size=1):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super(LISTA, self).__init__(We_shape, unroll_count,
                                    shrinkge_type, shared_threshold, batch_size)
        self._B = None
        if We is not None:
            #
            # warm start
            L = max(abs(np.linalg.eigvals(np.matmul(We, We.T))))
            self._theta = [tf.Variable(tf.constant(0.5/L, shape=[1, self.output_size], dtype=tf.float32), name='theta') for _ in range(unroll_count)]
            self._We = tf.Variable(We.T/L, name='We', dtype=tf.float32)
            self._S = tf.Variable(np.eye(self.output_size) - np.matmul(We, We.T/L),
                                  dtype=tf.float32)
            # self._Z = list()
        else:
            self._theta = [tf.Variable(tf.truncated_normal([1, self.output_size]),
                                       name='theta') for _ in range(unroll_count)]
            self._S = tf.Variable(tf.truncated_normal([self.output_size,
                                                       self.output_size]), name='S')

            self._We = tf.Variable(tf.truncated_normal([self.input_size,
                                                        self.output_size]),
                                   name='We', dtype=tf.float32)
 
    def _lista_step(self, Z, shrink_fn, theta):
        """ LISTA step.

        Args:
            Z:    sparse representation of last iteration.
            B:    result of B after last iteration.
            shrink_fn: type of shrinkage function to use. 
            S:         is not updated every unrolling rather via GD. 
        Returns:
            Z(t+1)
        """
        #
        # run one lcod pass through
        C = tf.add(self._B, tf.matmul(Z, self._S))
        Z = shrink_fn(C, theta)
        return Z

    def build_model(self):

        shrinkge_fn = self._shrinkge()
        self._B = tf.matmul(self._X, self._We)
        self._Z = shrinkge_fn(self._B, self._theta[0])
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            self._Z = self._lista_step(self._Z, shrinkge_fn, self._theta[t])

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_sum(tf.reduce_mean((self._Zstar -
                                                           self._Z) ** 2, 0))
                self._loss /= 2
                """
                self._loss += 0.01*tf.nn.l2_loss(self._theta)
                self._loss += 0.1*tf.nn.l2_loss(self._S)
                self._loss += 0.1*tf.nn.l2_loss(self._We) 
                """
        return self._loss

    
