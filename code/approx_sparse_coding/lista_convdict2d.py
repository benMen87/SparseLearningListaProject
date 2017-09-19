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
                 filter_arr=None, batch_size=1, kernel_count=36, kernel_size=[3, 5, 7],
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
        self._X = tf.placeholder(tf.float32, shape=(None, self.inputshape[0], self.inputshape[1], self.input_channels), name='X')
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
                self._theta = [tf.nn.relu(tf.Variable(tf.fill([1,
                     self.amount_of_kernals], value=0.1)), name='theta')] * unroll_count
            # Multi-resolution use 3 5 and 7 kernals
            self._amount_of_resolutions = 3
            init_We_3 = tf.nn.l2_normalize(tf.truncated_normal([3, 3,
                                          self.input_channels, self.amount_of_kernals //self._amount_of_resolutions]), dim=[0,1])
            init_We_5 = tf.nn.l2_normalize(tf.truncated_normal([5, 5,
                                          self.input_channels, self.amount_of_kernals // self._amount_of_resolutions]), dim=[0,1])
            init_We_7  = tf.nn.l2_normalize(tf.truncated_normal([7, 7,
                                          self.input_channels, self.amount_of_kernals // self._amount_of_resolutions]), dim=[0,1])


            self._We_3 = tf.Variable(0.1 * init_We_3, name='We3')
            self._We_5 = tf.Variable(0.1 * init_We_5, name='We5')
            self._We_7 = tf.Variable(0.1 * init_We_7, name='We7')


            self._Wd_3 = tf.Variable(tf.transpose(tf.reverse(self._We_3.initialized_value(), [0,1]), [0,1,3,2]), name='We3')
            self._Wd_5 = tf.Variable(tf.transpose(tf.reverse(self._We_5.initialized_value(), [0,1]), [0,1,3,2]), name='We5')
            self._Wd_7 = tf.Variable(tf.transpose(tf.reverse(self._We_7.initialized_value(), [0,1]), [0,1,3,2]), name='We7')

    def conv2d(self, _input, _kernel, name='conv2d'):
        res = tf.nn.conv2d(
                _input,
                _kernel,
                strides=[1, 1, 1, 1],
                padding='SAME', name=name
            )
        return res

    def multires_conv_we(self, _input, name="multi_res_we"):
        result = []
        We = [self._We_3, self._We_5, self._We_7]

        for ker in We:
            result.append(self.conv2d(_input, ker))
        
        result = tf.concat(result, axis=-1)  # from list to tensor
        return result


    def multires_conv_wd(self, _input, name="multi_res_wd"):
        Wd = [self._Wd_3, self._Wd_5, self._Wd_7]
        per_res_channels = self.amount_of_kernals // self._amount_of_resolutions
        result_resolution_list = []
        last_axis = len(_input.shape) - 1
        _input = tf.split(_input, [per_res_channel, per_res_channel, per_res_channel], axis=last_axis)  # from tensor to list

        for resolution_feature_map, ker in enumerate(_input, Wd):
            result_resolution_list.append(self.conv2d(resolution_feature_map, ker))
        
        result = tf.reduce_sum(result_resolution_list, axis=3)
        return result
            
    def build_model(self, mask=1):
        """
        mask - In case of inpainting etc.
        """
        shrinkge_fn = self._shrinkge()
        Wd = [self._Wd_3, self._Wd_5, self._Wd_7]

        X = tf.multiply(self._X, mask)
        B = self.multires_conv_we(X) 
        self._Z.append(shrinkge_fn(B, self._theta[0], '0'))

        #
        # run unrolling
        for t in range(1, self._unroll_count):
            conv_wd = multires_conv2d( self._Z[-1], Wd, concat_axis=-2, name='convWd')
            conv_wd = tf.multiply(conv_wd, mask)

            conv_we = self.multires_conv_we(conv_wd, name='convWe')
            res = self._Z[-1] - conv_we
            res_add_bias = res + B
            self._Z.append(shrinkge_fn(res_add_bias, self._theta[t], 'Z'+str(t)))

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
        return [self._Wd_3, self._Wd_5, self._Wd_7] 

    @property
    def We(self):
        return [self._We_3, self._We_5, self._We_7] 

    @property
    def output2d(self):
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
    def batch_size(self):
        return self.train_batch_size
