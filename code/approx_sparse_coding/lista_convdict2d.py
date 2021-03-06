"""
This module contanint Class LISTAConvDict2d
That an TF model for approximation CSC
"""
import sys
import tensorflow as tf
import numpy as np
from approx_sparse_coding.approx_sc import ApproxSC


class LISTAConvDict2d(ApproxSC):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """

    def __init__(self, inputshape, unroll_count, L,
                 filter_arr=None, batch_size=1, kernel_count=1, kernel_size=3,
                 shared_threshold=False, shrinkge_type='soft thresh',
                 init_params_dict={}, channel_size=3):
        """ Create a LCoD model.
        Args:
            inputshape: Input X is encoded using conv2D(We, X).
            unroll_size: Amount of times to repeat lcod block.
        """

        super(LISTAConvDict2d, self).__init__(inputshape, unroll_count, input2d=True,
                                              shrinkge_type=shrinkge_type, shared_threshold=shared_threshold,
                                              batch_size=batch_size, channel_size=channel_size)

        self.inputshape = inputshape
        self._X = tf.placeholder(tf.float32, shape=(None, None, None, self.input_channels), name='X')
        #
        # model variables
        if not init_params_dict and filter_arr is not None :
            self.kernel_size = filter_arr.shape[1]
            self.amount_of_kernals = filter_arr.shape[0]
            if self._shrinkge_type == 'soft thresh':
                self._theta = [
                    tf.nn.relu(tf.Variable(tf.random_uniform(maxval=0.5, shape=[1, self.patch_dim,
                    self.patch_dim, self.amount_of_kernals])), name='theta'+str(u))
                               for u in range(unroll_count)]
            else:
                raise NotImplementedError('shirnkge type not supported')

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

            if self._shrinkge_type == 'soft thresh':
                self._theta = [tf.nn.relu(tf.Variable(tf.fill([1,
                    self.amount_of_kernals], value=0.1)),
                                          name='theta')] * unroll_count
                # self._theta = [tf.Variable(init_params_dict['theta'][0], name='theta')]
                # self._theta += [tf.Variable(init_params_dict['theta'][-1], name='theta')
                #                 for _ in range(1, unroll_count)]
            else:
                raise NotImplementedError('shirnkge type not supported')

            self._Wd = tf.Variable(init_params_dict['Wd'],
                                   name='Wd', dtype=tf.float32)
            self._We = tf.Variable(init_params_dict['We'],
                                   name='We', dtype=tf.float32)
        else:
            self.amount_of_kernals = kernel_count
            self.kernel_size = kernel_size

            if self._shrinkge_type == 'soft thresh':
                #TODO: Notice thresh is now shared one for each feture map
                self._theta = tf.nn.relu(tf.Variable(tf.fill([1, self.amount_of_kernals],
                    0.05), name='theta'))
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

    def dilate_conv2d(_value, _filters, _name, _rate):
        res = tf.nn.atrous_conv2d(
                _value,
                _filters,
                rate=_rate,
                padding='SAME',
                name=_name
            )

    def conv2d(self, _value, _filters, _name, _dilate=False, _rate=[1, 2, 3]):
        if _dilate:
            dilate_conv2d(_value, _filters, _name, 3)
        else:
            res = tf.nn.conv2d(
                _value,
                _filters,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name=_name
            )
        return res

    def build_model(self, mask=None):
        """
        mask - In case of inpainting etc.
        """
        _Wd = tf.nn.l2_normalize(self._Wd, [0,1])
        _We = (0.1) * tf.nn.l2_normalize(self._We,
                [0,1])

        shrinkge_fn = self._shrinkge()
        mask = mask if mask is not None else tf.ones_like(self._X)

        X = tf.multiply(self._X, mask)
        B = self.conv2d(
            _value=X,
            _filters=_We,
            _name='bias'
            ) 
        self._Z.append(shrinkge_fn(B, self._theta, 'Z0'))
        #
        # run unrolling
        for t in range(1, self._unroll_count):
            conv_wd = self.conv2d(
                        _value=self._Z[-1],
                        _filters=_Wd,
                        _name='convWd'
                    )

            conv_wd = tf.multiply(conv_wd, mask)

            conv_we = self.conv2d(
                        _value=conv_wd,
                        _filters=_We,
                        _name='convWe'
                    )
            res = self._Z[-1] - conv_we
            res_add_bias = res + B
            self._Z.append(shrinkge_fn(res_add_bias, self._theta, 'Z'+str(t)))

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
    def output(self):
        """
        returns array of 2d feature maps
        """
        return self._Z[-1]

    def output2d_i(self, i):
        """
        return output Z
        """
        _Z = self._Z[i]

        return _Z


    @property
    def input2D(self):
        return self._X

    @property
    def output1d(self):
        return tf.reshape(tf.transpose(self._Z[-1], [0, 3, 1, 2]),
                          [-1, self.input_size * self.amount_of_kernals])

    @property
    def batch_size(self):
        return self.train_batch_size
