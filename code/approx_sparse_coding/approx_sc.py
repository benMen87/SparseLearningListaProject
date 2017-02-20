import tensorflow as tf
import numpy as np

class ApproxSC(object):
    """description of class"""
    def __init__(self, We_shape, unroll_count,  shrinkge_type='soft thresh'):
        """ Create a LCoD model.
        
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
        self._Z     = None
        #
        # Trainable Parameters 
        self._theta = None
        self._S     = None
        self._We    = None
        #
        # Loss
        self._loss  = None


    def build_model(self):
        assert('Abstract class define child loss')

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            return self._soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _soft_thrsh(self, B):
        soft_thrsh_out = tf.multiply(tf.sign(B), tf.nn.relu(tf.subtract(tf.abs(B), self._theta)))
        #soft_thrsh_out  = tf.nn.relu(tf.divide(B, self._theta) - 1)
        #soft_thrsh_out  = tf.multiply(tf.sign(B), soft_thrsh_out)
        #soft_thrsh_out  = tf.multiply(soft_thrsh_out, self._theta)
        return soft_thrsh_out

    def _double_tanh(self, B):
        """
        not implemented 
        """
        raise NotImplementedError('Double Tanh not implemented')


    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        assert('Abstract class define child loss')

    @property
    def output(self):
        if self._Z is None:
            assert('Abstract class define child Z') 
        return self._Z

    @property
    def input(self):
        return self._X

    @property
    def target(self):
        return self._Zstar

    @property
    def theta(self):
        if self._theta is None:
            assert('Abstract class define child theta') 

        return self._theta
        
    @property
    def S(self):
        if self._S is None:
            assert('Abstract class define child S') 
        return self._S

    @property
    def We(self):
        if self._We is None:
            assert('Abstract class define child We') 
        return self._We

    @property
    def unroll_count(self):
        return self._unroll_count




