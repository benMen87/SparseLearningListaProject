import tensorflow as tf
import numpy as np
import approx_sparselearnig_dataset as data_handelr

class Lcod(object):
    """tensorflow implementation of lcod from
       Learning Fast Approximations of Sparse Coding - http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf
    """

    def __init__(self, We_shape, _unroll_count, _shrinkge_type='soft thresh'):
        """ Create a Lcod model.
        
        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        
        self._We_shape      = We_shape
        self._unroll_count = _unroll_count
        self._shrinkge_type = _shrinkge_type
        #
        # graph i/o
        self._X     = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1),
                                      name='X')
        self._Zstar = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1),
                                       name='Zstar')
        self._Z    = None
        self._loss = None

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            return self._soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _soft_thrsh(self, B):
        with tf.variable_scope('h_theta'):
            theta = tf.get_variable(name='theta', dtype=tf.float32, initializer=np.constant_initializer(0.5))
        soft_thrsh_out = tf.multiply(tf.sign(tf.nn.relu(tf.subtract(tf.abs(B), theta))))
        return soft_thrsh_out

    def _double_tanh(self, B):
        """
        not implemented 
        """
        raise NotImplementedError('Double Tanh not implemented')

    def _lcod_step(self, Z_prev, B_prev, S, shrink_fn):
        """ Lcod step.

        Args:
            Z_prev:    sparse representation of last iteration.
            B_prev:    result of B after last iteration.
            shrink_fn: type of shrinkage function to use. 
            S:         is not updated every unrolling rather via GD. 
        Returns:
            update Z_prev and B_prev
        """
        #
        # run one lcod pass through
        Z_hat     = shrink_fn(B_prev)
        res       = tf.subtract(Z_hat, Z_prev)
        k         = tf.arg_max(np.abs(res))
        B_prev    = np.add(B_prev, np.multiply(S[:, k], res))
        Z_prev[k] = Z_hat[k]                        

        return (Z_prev, B_prev)

    def model(self, amount_unroll=7, train=True):
        m, n = self._We_shape
        shrinkge_fn = self._shrinkge()

        with tf.variable_scope('lcod_model'):
            #
            # lcod output - Z is learned via lcod algorithm NOT to update via GD!!!!
            Z  = tf.Variable(tf.zeros_initializer(shape=(m, 1)), trainable=False)
            S  = tf.Variable( tf.random_normal_initializer(shape=(m, m), name='S'))
            We = tf.Variable( tf.random_normal_initializer(shape=(m,n), name='We'))

        B = tf.matmul(We, self._X)
        #
        # run unrolling
        Z_arr = []
        B_arr = []
        for t in  amount_unroll:
            Z, B = self._lcod_step(Z, B, S, shrinkge_fn)
            Z_arr.append(Z)
            B_arr.append(B)

        self._Z = shrinkge_fn(B)

        return Z


    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_mean((self.targets - self.predictions)**2, name='loss')
                
        return self._loss

    @property
    def Z(self):
        if self._Z is None:
            raise EnvironmentError('must run model first')
        return self._Z
