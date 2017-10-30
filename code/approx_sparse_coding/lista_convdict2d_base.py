"""
This module contanint Class LISTAConvDict2dBase
That an TF model for approximation CSC
"""

import sys
import tensorflow as tf
import numpy as np
from approx_sparse_coding.approx_sc import ApproxSC

class LISTAConvDict2dBase(object):
    
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """

    def __init__(self, unroll_count,
                 inputshape=None,
                 filter_arr=None,
                 batch_size=None,
                 channel_size=3,
                 shrinkge_type='soft_thresh',
                 **kwargs ):
        """ Create a LISTAConv model.
        Args:
            inputshape: Input X is encoded using conv2D(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """

        self.inputshape = inputshape
        self.batch_size = batch_size
        self.input_channels = input_channels
        self._shrinkge_type = shrinkge_type 
        self._X = tf.placeholder(tf.float32, shape=(self.batch_size, self.inputshape, self.inputshape, self.input_channels), name='X')
        self._mask  = tf.placeholder_with_default(1, shape=self._X.shape, name='mask')
        #
        # model variables
        init_type = kwargs.get('init_type', 'random')
        if init_type == 'filter_arr' :
            self.init_from_filter_array(kwargs['filter_arr'])
        elif init_type == 'param_dict':
            self.init_from_param_dict(kwargs['init_params_dict'])
        elif init_type == 'random':
            self.init_random_ista_coherent(self, kwargs)
        else:
            raise ValueError('init type: {} is not reconized'.format(init_type))

    def init_random_ista_coherent(self, kwargs):
        """
        All values are initlized randomly but with respect to their ISTA connections. 
        """
        self.amount_of_kernals = kwargs.get('kernel_count', 64)
        self.kernel_size = kwargs.get('kernel_size', 7)

        if self._shrinkge_type == 'soft thresh':
            #TODO: Notice thresh is now shared one for each feture map
            thrsh = kwargs.get('init_threshold', 0.1)
            self._theta = tf.nn.relu(tf.Variable(tf.fill([thrsh, self.amount_of_kernals],
                0.1), name='theta'))
          #       self.amount_of_kernals], value=0.01)), name='theta')] * unroll_count
        elif self._shrinkge_type == 'smooth soft thresh':
            beta = [tf.Variable(tf.fill([1, self.amount_of_kernals], 5.0), name='beta'+str(u))
                          for u in range(unroll_count)]
            b = [tf.Variable(tf.fill([1, self.amount_of_kernals], 0.5), name='b'+str(u))
                       for u in range(unroll_count)]
            self._theta = zip(beta, b)

        init_We = tf.nn.l2_normalize(tf.truncated_normal([self.kernel_size, self.kernel_size,
                                      self.input_channels, self.amount_of_kernals]), dim=[0,1])
        self._We = tf.Variable(init_We, name='We')
        self._Wd = tf.Variable(tf.transpose(tf.reverse(self._We.initialized_value(), [0,1]), [0,1,3,2]), name='We')

    def init_from_param_dict(self, init_params_dict):
        """
        All inilized values are given via init_params_dict
        """
            self.kernel_size = init_params_dict['Wd'].shape[1]
            self.amount_of_kernals = init_params_dict['Wd'].shape[2]

            if self._shrinkge_type == 'soft thresh':
                self._theta = tf.nn.relu(tf.Variable(tf.fill([1,
                    self.amount_of_kernals], value=0.1)), name='theta')
            else:
                raise NotImplementedError('shirnkge type not supported')

            self._Wd = tf.Variable(init_params_dict['Wd'],
                                   name='Wd', dtype=tf.float32)
            self._We = tf.Variable(init_params_dict['We'],
                                   name='We', dtype=tf.float32)

    def init_from_filter_array(self, filter_arr):
        """
        inilize values from given filter array
        """
        self.kernel_size = filter_arr.shape[1]
        self.amount_of_kernals = filter_arr.shape[0]

        if self._shrinkge_type == 'soft thresh':
            self._theta = tf.nn.relu(tf.Variable(tf.fill([1, self.amount_of_kernals],
                    0.05), name='theta'))
        else:
            raise NotImplementedError('shirnkge type not supported')

        flipfilter_arr = np.array([np.flip(np.flip(f,0),1) for f in filter_arr])
        flipfilter_arr = np.expand_dims(flipfilter_arr, axis=-1)
        flipfilter_arr = np.transpose(flipfilter_arr, [1,2,3,0])
        filter_arr = np.expand_dims(filter_arr, axis=-1)
        filter_arr = np.transpose(filter_arr, [1,2,0,3])
        self._We = (1/L)*tf.Variable(flipfilter_arr, name='We', dtype=tf.float32)
        self._Wd = tf.Variable(filter_arr, name='Wd', dtype=tf.float32)

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

    def build_model(self, mask=None):
        """
        mask - In case of inpainting etc.
        """
        shrinkge_fn = self._shrinkge()
        _Wd = tf.nn.l2_normalize(self._Wd, [0,1])
        _We = (0.1) * tf.nn.l2_normalize(self._We, [0,1])


        X = tf.multiply(self._X, self._mask)
        B = self.conv2d_enc(
            _value=X,
            _name='bias'
            ) 
        self._Z = shrinkge_fn(B, self._theta, 'Z_0')
        tf.add_to_collection('SC_Zt', self._Z)
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            conv_wd = self.conv2d_dec(
                _value=self._Z,
                _name='convWd'
                )
            conv_wd = tf.multiply(
               conv_wd,
               self._mask
               )
            conv_we = self.conv2d_enc(
               _value=conv_wd,
               _name='convWe'
               )
            res = self._Z - conv_we
            res_add_bias = res + B
            self._Z = shrinkge_fn(
                res_add_bias,
                self._theta,
                'Z_'+str(t)
                )
            tf.add_to_collection('SC_Zt', self._Z)

    def output_shape(self):
        return self._Z.get_shape().as_list()

    @property
    def Wd(self):
        return self._Wd

    @property
    def We(self):
        return self._We

    @property
    def output(self):
        """
        returns array of 2d feature maps
        """
        return self._Z

    def output2d_t(self, i):
        """
        return output Z
        """
        _Zt = tf.get_collection('SC_Zt')[i]
        return _Zt

    @property
    def input(self):
        return self._X

    @property
    def output1d(self):
        return tf.reshape(tf.transpose(self._Z, [0, 3, 1, 2]),
                          [-1, self.input_size * self.amount_of_kernals])

    @property
    def batch_size(self):
        return self._X.get_shape().as_list()[0]
