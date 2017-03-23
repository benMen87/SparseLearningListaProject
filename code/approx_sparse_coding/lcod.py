import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from Utils import db_tools


class LCoD(ApproxSC):
    """tensorflow implementation of lcod from
       Learning Fast Approximations of Sparse Coding -
       http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf
    """

    def __init__(self, We_shape, unroll_count, We=None,
                 shrinkge_type='soft thresh', batch_size=1):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        if batch_size != 1:
            raise NotSupportedErr('LCoD Currently only supports\
                                   batch size of 1')
        super(LCoD, self).__init__(We_shape, unroll_count,
                                   shrinkge_type, batch_size)

        self._Z = tf.zeros([1, self.output_size])
        if We is not None:
            # warm start
            self._theta = tf.Variable(tf.constant(0.5, shape=[1, self.output_size]),
                                      name='theta')
            self._We = tf.Variable(We.T, name='We', dtype=tf.float32)
            self._S = tf.Variable(np.eye(self.input_size) - np.matmul(We.T, We),
                                  dtype=tf.float32)
        else:
            self._theta = tf.Variable(tf.truncated_normal([1, self.output_size]),
                                      name='theta')
            self._S = tf.Variable(tf.truncated_normal([self.output_size, self.output_size]), name='S')
            self._We = tf.Variable(tf.truncated_normal([self.input_size, self.output_size]), name='We',
                                   dtype=tf.float32)

    def _lcod_step(self, Z, B, S, shrink_fn):
        """ LCoD step.

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
        Z_hat = shrink_fn(B)
        res = tf.subtract(Z_hat, Z)
        k = tf.to_int32(tf.argmax(tf.abs(res), axis=1))
        #
        # update
        # TODO HS: check this out if problems
        B = tf.add(B, tf.multiply(tf.gather(S, k),
                                  tf.gather(tf.transpose(res), k)))

        unchanged_indices = tf.range(tf.size(Z))
        change_indices = k
        Z = tf.dynamic_stitch([unchanged_indices, change_indices],
                              [tf.transpose(Z), tf.gather(tf.transpose(Z_hat), k)])
        Z = tf.reshape(Z, Z_hat.get_shape())
        return (Z, B)

    def build_model(self):
        shrinkge_fn = self._shrinkge()

        B = tf.matmul(self._X, self._We)
        #
        # run unrolling
        for t in range(self._unroll_count):
            self._Z, B = self._lcod_step(self._Z, B, self._S, shrinkge_fn)
        self._Z = shrinkge_fn(B)

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.nn.l2_loss(self._Zstar - self._Z, name='loss')
                # self._loss += 1e6*tf.nn.l2_loss(tf.minimum(self._theta, 0))  
                # self._loss += tf.nn.l2_loss(self._theta)
                # self._loss += 0.1*tf.nn.l2_loss(self._S)
                # self._loss += 0.1*tf.nn.l2_loss(self._We) 
        return self._loss

