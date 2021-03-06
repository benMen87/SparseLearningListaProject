import tensorflow as tf
import numpy as np


class ApproxSC(object):
    """description of class"""
    def __init__(self, We_shape, unroll_count, input2d = False,
                 shrinkge_type='soft thresh', shared_threshold=False, batch_size=1, channel_size=1):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """
        self.train_batch_size = np.int32(batch_size)
        self._We_shape = We_shape
        self._unroll_count = unroll_count
        self._shrinkge_type = shrinkge_type
        self.shared_threshold = shared_threshold
        self.input_channels = channel_size
        #
        # graph i/o
        b_dim = self.train_batch_size
        b_dim = None 
        if input2d:
            # dim_size = np.int(np.sqrt(self.input_size))
            # self._X = tf.placeholder(tf.float32, shape=(b_dim, dim_size, dim_size, self.input_channels),
            #                          name='X')
            pass
        else:
            self._X = tf.placeholder(tf.float32, shape=(b_dim, self.input_size),
                                     name='X')

        self._Zstar = tf.placeholder(tf.float32,
                                     shape=(b_dim, self.output_size),
                                     name='Zstar')
        self._Z = []
        #
        # Trainable Parameters
        self._last_theta = None
        self._S = None
        self._We = None
        #
        # Loss
        self._loss = None

    def build_model(self):
        assert('Abstract class define child loss')

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            return self._soft_thrsh
        elif self._shrinkge_type == 'smooth soft thresh':
            return self._smooth_soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _smooth_soft_thrsh(self, X, theta, name=''):
        """
        X  - Input
        theta - tuple(beta, b)
        beta controls the smoothness of the kink of shrinkage operator,
        and b controls the location of the kink
        """
        beta, b = theta
        
        def smooth_relu(x, beta, b, name):
            first = beta * b * tf.ones_like(x)
            second = beta * x
            third = tf.zeros_like(x)
            # TODO: logsum exp works well for overflow but seems to cause underflow as well
            return (1/beta)*tf.reduce_logsumexp([first, second, third], 0, name=name) - b

        return smooth_relu(X, beta, b, name+'_right') - smooth_relu(-X, beta, b, name+'_left')

    def _soft_thrsh(self, X, theta, name=''):
        return tf.subtract(tf.nn.relu(X-theta), tf.nn.relu(-X-theta), name=name)

    def _double_tanh(self, X):
        raise NotImplementedError('Double Tanh not implemented')

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        assert 'Abstract class define child loss'

    @property
    def output(self):
        """
        return output Z
        """
        if self._Z is None:
            assert('Abstract class define child Z')
        return self._Z

    @property
    def input(self):
        return self._X

    @property
    def target(self):
        return self._Zstar

    def _theta(self):
            assert('Abstract class define child theta')

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
    def input_size(self):
        return self._We_shape[1]

    @property
    def output_size(self):
        return self._We_shape[0]

    @property
    def unroll_count(self):
        return self._unroll_count




