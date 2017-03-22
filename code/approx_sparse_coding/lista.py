import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC


class LISTA(ApproxSC):
    """description of class"""

    def __init__(self, We_shape, unroll_count, We=None,
                 shrinkge_type='soft thresh', batch_size=1):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super(LISTA, self).__init__(We_shape, unroll_count,
                                    shrinkge_type, batch_size)
        m, n = self._We_shape
        if We is not None:
            #
            # warm start
            L = max(abs(np.linalg.eigvals(np.matmul(We, We.T))))
            self._theta = tf.Variable(tf.constant(0.5/L, shape=[1, m],
                                      dtype=tf.float32), name='theta')
            self._We = tf.Variable(We.T/L, name='We', dtype=tf.float32)
            self._S = tf.Variable(np.eye(m) - np.matmul(We.T/L, We),
                                  dtype=tf.float32)
        else:
            self._theta = tf.Variable(tf.truncated_normal([1, m]),
                                      name='theta')
            self._S = tf.Variable(tf.truncated_normal([m, m]), name='S')
            self._We = tf.Variable(tf.truncated_normal([n, m]),
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
        C = tf.add(B, tf.matmul(Z, S))
        Z = shrink_fn(C)
        return (Z, B)

    def build_model(self):

        shrinkge_fn = self._shrinkge()
        B = tf.matmul(self._X, self._We)
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
                self._loss = tf.nn.l2_loss(self._Zstar - self._Z, name='loss')
                self._loss /= self.train_batch_size
                """
                self._loss += 0.01*tf.nn.l2_loss(self._theta)
                self._loss += 0.1*tf.nn.l2_loss(self._S)
                self._loss += 0.1*tf.nn.l2_loss(self._We) 
                """
        return self._loss
