import tensorflow as tf
import numpy as np
import os
import sys
from approx_sc import ApproxSC

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from Utils import db_tools

class LCoD(ApproxSC):
    """tensorflow implementation of lcod from
       Learning Fast Approximations of Sparse Coding - http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf
    """

    def __init__(self, We_shape, unroll_count, We=None,  shrinkge_type='soft thresh'):
        """ Create a LCoD model.
        
        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super().__init__(We_shape, unroll_count, We, shrinkge_type)
        m, n = self._We_shape
        self._Z     = tf.Variable(tf.zeros_initializer([m, 1]), trainable=False)
     
 
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
        Z_hat     = shrink_fn(B)
        res       = tf.subtract(Z_hat, Z)
        k         = tf.to_int32(tf.argmax(tf.abs(res), axis=0))
        #
        # update
        B = tf.add(B, tf.multiply(tf.transpose(tf.gather(tf.transpose(S), k)), tf.gather(res, k)))

        unchanged_indices = tf.range(tf.size(Z))
        change_indices    = k
        Z = tf.dynamic_stitch([unchanged_indices, change_indices], [Z, tf.gather(Z_hat, k)])
        Z = tf.reshape(Z, Z_hat.get_shape())

        return (Z, B)

    def build_model(self):
        shrinkge_fn = self._shrinkge()

        B = tf.matmul(self._We, self._X)
        #
        # run unrolling
        for t in  range(self._unroll_count):
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
                #self._loss += 0.01*tf.nn.l2_loss(self._theta)
                #self._loss += 0.1*tf.nn.l2_loss(self._S)
                #self._loss += 0.1*tf.nn.l2_loss(self._We) 
        return self._loss

