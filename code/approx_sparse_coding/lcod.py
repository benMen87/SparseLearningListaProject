import tensorflow as tf
import numpy as np
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from Utils import db_tools

class Lcod(object):
    """tensorflow implementation of lcod from
       Learning Fast Approximations of Sparse Coding - http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf
    """

    def __init__(self, We_shape, unroll_count, shrinkge_type='soft thresh'):
        """ Create a Lcod model.
        
        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """

        self._We_shape      = We_shape
        self._unroll_count  = unroll_count
        self._shrinkge_type = shrinkge_type

        m, n = self._We_shape
        #
        # graph i/o
        self._X     = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1), name='X')
        self._Zstar = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1), name='Zstar')
        self._Z     = tf.Variable(tf.zeros_initializer([m, 1]), trainable=False)
        #
        # Trainable Parameters 
        self._S     = tf.Variable( tf.truncated_normal([m, m]), name='S')
        self._We    = tf.Variable( tf.truncated_normal([m,n]), name='We')
        self._theta = tf.Variable(0.5, name='theta')
        #
        # Loss
        self._loss  = None

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            return self._soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _soft_thrsh(self, B):
        soft_thrsh_out = tf.multiply(tf.sign(B), tf.nn.relu(tf.subtract(tf.abs(B), self._theta)))
        return soft_thrsh_out

    def _double_tanh(self, B):
        """
        not implemented 
        """
        raise NotImplementedError('Double Tanh not implemented')

    def _lcod_step(self, Z, B, S, shrink_fn):
        """ Lcod step.

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
        k         = tf.to_int32(tf.argmax(np.abs(res), axis=0))
        #
        # update
        B = np.add(B, np.multiply(tf.transpose(tf.gather(tf.transpose(S), k)), tf.gather(res, k)))

        unchanged_indices = tf.range(tf.size(Z))
        change_indices    = k
        Z = tf.dynamic_stitch([unchanged_indices, change_indices], [Z, tf.gather(Z_hat, k)])
        Z = tf.reshape(Z, Z_hat.get_shape())

        return (Z, B)

    def build_model(self, amount_unroll=7):
        shrinkge_fn = self._shrinkge()

        print('We shape:{}, x shape: {}'.format(self._We.get_shape(), self._X.get_shape()))
        B = tf.matmul(self._We, self._X)
        print('B  shape: {}'.format(B.get_shape()))
        #
        # run unrolling
        #Z_arr = []
        #B_arr = [] 
        for t in  range(amount_unroll):
            self._Z, B = self._lcod_step(self._Z, B, self._S, shrinkge_fn)
        self._Z = shrinkge_fn(B)
        #return Z

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_mean((self._Zstar - self._Z)**2, name='loss')         
        return self._loss

    @property
    def output(self):
       return self._Z
    @property
    def input(self):
        return self._X

    @property
    def target(self):
        return self._Zstar

    @property
    def theta(self):
        if hasattr(self, '_theta'):
            return self._theta
        else:
            return None
    @property
    def S(self):
        return self._S

    @property
    def Wd(self):
        return self._Wd
   
