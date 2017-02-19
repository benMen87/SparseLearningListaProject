import tensorflow as tf
import numpy as np
import os
import sys
from approx_sc import ApproxSC


class LISTA(ApproxSC):
    """description of class"""

    def __init__(self, We_shape, unroll_count, We=None,  shrinkge_type='soft thresh'):
        """ Create a LCoD model.
        
        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        super().__init__(We_shape, unroll_count, We, shrinkge_type)
 
    def _lista_step(self, Z, B, S, shrink_fn):
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
        C = tf.add(B, tf.matmul(S, Z))
        Z = shrink_fn(C)
        return (Z, B)

    def build_model(self):

        shrinkge_fn = self._shrinkge()
        B = tf.matmul(self._We, self._X)
        self._Z = shrinkge_fn(B)
        #
        # run unrolling
        for t in  range(self._unroll_count):
            self._Z, B = self._lista_step(self._Z, B, self._S, shrinkge_fn)

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
