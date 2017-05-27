import os
import sys
import tensorflow as tf
import numpy as np
from approx_sc import ApproxSC


class LISTAConvDict (ApproxSC):
    """Class of approximate SC based on convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """

    def __init__(self, We_shape, unroll_count, filter_arr,
                 L, batch_size=1, kernal_size=3,
                 shared_threshold=False, shrinkge_type='soft thresh',
                 init_params_dict={}):
        """ Create a LCoD model.

        Args:
            We_shape: Input X is encoded using matmul(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """

        super(LISTAConvDict, self).__init__(We_shape, unroll_count,
                                            shrinkge_type, shared_threshold, batch_size)

        self.input_channels = 1  # TODO: add support for RGB?

        #
        # model variables
        if not init_params_dict:
            self.kernal_size = filter_arr.shape[1]
            self.amount_of_kernals = filter_arr.shape[0]
            self._theta = [tf.Variable(tf.constant(0.5/L,
                                       shape=[1, self.input_size, self.amount_of_kernals],
                           dtype=tf.float32), name='theta')
                           for _ in range(unroll_count)]
            transpose_filt = np.array([f[::-1] for f in filter_arr])
            self._Wd = tf.Variable(np.expand_dims(transpose_filt.T, axis=-1),
                                   name='Wd', dtype=tf.float32)
            self._We = (1/L)*tf.Variable(np.expand_dims(filter_arr.T, axis=1),
                                         name='We', dtype=tf.float32)
        else:
            self.kernal_size = init_params_dict['Wd'].shape[0]
            self.amount_of_kernals = init_params_dict['Wd'].shape[1]
            self._theta = [tf.Variable(init_params_dict['theta'][0], name='theta')]
            self._theta += [tf.Variable(init_params_dict['theta'][-1], name='theta')
                            for _ in range(1, unroll_count)]
            self._Wd = tf.Variable(init_params_dict['Wd'],
                                   name='Wd', dtype=tf.float32)
            self._We = tf.Variable(init_params_dict['We'],
                                   name='We', dtype=tf.float32)

    def build_model(self):
        shrinkge_fn = self._shrinkge()

        B = tf.nn.conv1d(tf.expand_dims(self._X, axis=-1),
                         self._We, stride=1,
                         padding='SAME', name='bias')
        self._Z.append(shrinkge_fn(B, self._theta[0], 'Z0'))
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            conv_wd = tf.nn.conv1d(self._Z[t-1], self._Wd, stride=1,
                                   padding='SAME', name='convWd')

            conv_we = tf.nn.conv1d(conv_wd, self._We, stride=1,
                                   padding='SAME', name='convWe')
            res = self._Z[t-1] - conv_we
            res_add_bias = res + B

            self._Z.append(shrinkge_fn(res_add_bias, self._theta[t], 'Z{}'.format(t)))

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

    @property
    def Wd(self):
        return self._Wd

    @property
    def output(self):
        return tf.reshape(tf.transpose(self._Z[-1], [0, 2, 1]),
                          [-1, self.input_size * self.amount_of_kernals])

    @property
    def batch_size(self):
        return self.train_batch_size
