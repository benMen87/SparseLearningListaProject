import tensorflow as tf
import numpy as np
import approx_sparselearnig_dataset as data_handelr

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

        #
        # graph i/o
        self._X     = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1), name='X')
        self._Zstar = tf.placeholder(tf.float32, shape=(self._We_shape[1], 1), name='Zstar')
        self._Z    = None
        self._loss = None

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            with tf.variable_scope('h_theta'):
                theta = tf.get_variable(name='theta', shape=(1,), dtype=tf.float32, initializer=tf.constant_initializer(0.5))
            return self._soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _soft_thrsh(self, B):
        with tf.variable_scope('h_theta', reuse=True):
            theta = tf.get_variable(name='theta')
        soft_thrsh_out = tf.multiply(tf.sign(B), tf.nn.relu(tf.subtract(tf.abs(B), theta)))
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
            update Z and B
        """
        #
        # run one lcod pass through
        Z_hat     = shrink_fn(B_prev)
        res       = tf.subtract(Z_hat, Z_prev)
        k         = tf.to_int32(tf.argmax(np.abs(res), axis=0))
        #
        # update
        B = np.add(B_prev, np.multiply(tf.transpose(tf.gather(tf.transpose(S), k)), tf.gather(res, k)))

        unchanged_indices = tf.range(tf.size(Z_prev))
        change_indices    = k
        Z = tf.dynamic_stitch([unchanged_indices, change_indices], [Z_prev, tf.gather(Z_hat, k)])
        Z = tf.reshape(Z, Z_prev.get_shape())

        return (Z, B)

    def build_model(self, amount_unroll=7):
        m, n = self._We_shape
        shrinkge_fn = self._shrinkge()

        with tf.variable_scope('lcod_model'):
            #
            # lcod output - Z is learned via lcod algorithm NOT to update via GD!!!!
            Z  = tf.Variable(tf.zeros_initializer([m, 1]), trainable=False)
            S  = tf.Variable( tf.truncated_normal([m, m]), name='S')
            We = tf.Variable( tf.truncated_normal([m,n]), name='We')

        print('We shape:{}, x shape: {}'.format(We.get_shape(), self._X.get_shape()))
        B = tf.matmul(We, self._X)
        print('B  shape: {}'.format(B.get_shape()))
        #
        # run unrolling
        Z_arr = []
        B_arr = []
        for t in  range(amount_unroll):
            Z, B = self._lcod_step(Z, B, S, shrinkge_fn)
            Z_arr.append(Z)
            B_arr.append(B)
        self._Z = shrinkge_fn(B)

        return Z

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_mean((self.target - self.output)**2, name='loss')         
        return self._loss

    @property
    def output(self):
        if self._Z is None:
            raise EnvironmentError('must run model first')
        return self._Z

    @property
    def input(self):
        return self._X

    @property
    def target(self):
        return self._Zstar

