import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC


class LISTAConvDict2d (ApproxSC):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """

    def __init__(self, We_shape, unroll_count, L, 
                 filter_arr=None, batch_size=1, kernel_count=1, kernel_size=3,
                 shared_threshold=False, shrinkge_type='soft thresh',
                 init_params_dict={}):
        """ Create a LCoD model.
        Args:
            We_shape: Input X is encoded using conv2D(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """

        super(LISTAConvDict2d, self).__init__(We_shape, unroll_count,
                                              shrinkge_type, shared_threshold, batch_size)

        self.input_channels = 1  # TODO: add support for RGB?
        self.patch_dim = np.int(np.sqrt(self.input_size))
        #
        # model variables
        if not init_params_dict and filter_arr is not None :
            self.kernel_size = filter_arr.shape[1]
            self.amount_of_kernals = filter_arr.shape[0]
            self._theta = [tf.Variable(tf.constant(0.5/L,
                                       shape=[1, self.patch_dim, self.patch_dim, self.amount_of_kernals],
                           dtype=tf.float32), name='theta')
                           for _ in range(unroll_count)]

            flipfilter_arr = np.array([np.flip(np.flip(f,0),1) for f in filter_arr])
            flipfilter_arr = np.expand_dims(flipfilter_arr, axis=-1)
            flipfilter_arr = np.transpose(flipfilter_arr, [1,2,3,0])
            filter_arr = np.expand_dims(filter_arr, axis=-1)
            filter_arr = np.transpose(filter_arr, [1,2,0,3])
            self._We = (1/L)*tf.Variable(flipfilter_arr, name='We', dtype=tf.float32)
            self._Wd = tf.Variable(filter_arr, name='Wd', dtype=tf.float32)
            
        elif init_params_dict:
            self.kernel_size = init_params_dict['Wd'].shape[1]
            self.amount_of_kernals = init_params_dict['Wd'].shape[2]
            self._theta = [tf.Variable(init_params_dict['theta'][0], name='theta')]
            self._theta += [tf.Variable(init_params_dict['theta'][-1], name='theta')
                            for _ in range(1, unroll_count)]
            self._Wd = tf.Variable(init_params_dict['Wd'],
                                   name='Wd', dtype=tf.float32)
            self._We = tf.Variable(init_params_dict['We'],
                                   name='We', dtype=tf.float32)
        else:
            self.amount_of_kernals = kernel_count
            self.kernel_size = kernel_size
            self._theta = [tf.Variable(tf.fill([1, self.patch_dim, self.patch_dim, self.amount_of_kernals], 0.2), name='theta'+str(u))
                           for u in range(unroll_count)]
            init_We = tf.truncated_normal([self.kernel_size, self.kernel_size,
                                          self.input_channels, self.amount_of_kernals], stddev=0.05)
            init_Wd = tf.truncated_normal([self.kernel_size, self.kernel_size,
                                           self.amount_of_kernals, self.input_channels], stddev=0.05)
            self._We = tf.Variable(init_We, name='We')
            self._Wd = tf.Variable(init_Wd, name='Wd')

    def build_model(self):
        shrinkge_fn = self._shrinkge()

        B = tf.nn.conv2d(tf.reshape(self._X, [-1, self.patch_dim, self.patch_dim, 1]),
                         self._We, strides=[1, 1, 1, 1],
                         padding='SAME', name='bias')
        self._Z = shrinkge_fn(B, self._theta[0], 'Z0')
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            conv_wd = tf.nn.conv2d(self._Z, self._Wd,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME', name='convWd')

            conv_we = tf.nn.conv2d(conv_wd, self._We,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME', name='convWe')
            res = self._Z - conv_we
            res_add_bias = res + B

            self._Z = shrinkge_fn(res_add_bias, self._theta[t], 'Z'+str(t))

    @property
    def loss(self):
        """ A 0-D float32 Tensor.
            function is used as a limited setter and a getter...
        """
        if self._loss is None:
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_sum(tf.reduce_mean((self.output -
                                                           self.target) ** 2, 0))
                self._loss /= 2
        return self._loss

    def output_shape(self):
        return self._Z.get_shape()

    @property
    def Wd(self):
        return self._Wd

    @property
    def output2D(self):
        return self._Z
    
    @property
    def input2D(self):
        return tf.reshape(self._X, [-1, self.patch_dim, self.patch_dim, 1])

    @property
    def output(self):
        return tf.reshape(tf.transpose(self._Z, [0, 3, 1, 2]),
                          [-1, self.input_size * self.amount_of_kernals])

    @property
    def output2D(self):
        return self._Z

    @property
    def batch_size(self):
        return self.train_batch_size
