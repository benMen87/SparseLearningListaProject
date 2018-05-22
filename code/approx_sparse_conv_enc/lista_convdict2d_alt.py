import tensorflow as tf
import numpy as np
from lista_convdict2d import LISTAConvDict2d

class LISTAConvDict2dAlt(LISTAConvDict2d):
    def __init__(
            self, unroll_count,
            inputshape=None,
            batch_size=None,
            input_channels=1,
            **kwargs
            ):

        kwargs['inputshape'] = inputshape
        kwargs['batch_size'] = batch_size
        kwargs['input_channels'] = input_channels
        kwargs['shrinkge_type'] = kwargs.get('shrinkge_type', 'soft thresh')


        super(LISTAConvDict2d, self).__init__(
            unroll_count=unroll_count, **kwargs
            )


    def init_random_ista_coherent(self, kwargs):
        """
        All values are initlized randomly but with respect to their ISTA connections. 
        """
        self.amount_of_kernals = kwargs.get('kernel_count', 64)
        self.kernel_size = kwargs.get('kernel_size', 7)

        thrsh = kwargs.get('init_threshold', 0.1)
        self._theta = tf.nn.relu(tf.Variable(tf.fill([1, self.amount_of_kernals],
                thrsh), name='theta'))
          #       self.amount_of_kernals], value=0.01)), name='theta')] * unroll_count

        init_We = tf.nn.l2_normalize(tf.truncated_normal([self.kernel_size, self.kernel_size,
                                      self.input_channels, self.amount_of_kernals]), dim=[0,1])
        init_Wd = tf.nn.l2_normalize(tf.truncated_normal([self.kernel_size, self.kernel_size,
                                      self.amount_of_kernals, self.amount_of_kernals]), dim=[0,1])
        self._We = tf.Variable(0.1 * init_We, name='We')
        self._Wd = tf.Variable(init_Wd, name='Wd')

    def build_model(self, inputs):
        """
        mask - In case of inpainting etc.
        """
        shrinkge_fn = self._shrinkge()
        self._creat_mask(tf.shape(inputs))

        #X = self._apply_mask(inputs)
        X = inputs # TODO: find a dynamic way to choose apply mask
        B = self._conv2d_enc(
            _val=X,
            _name='bias'
            ) 
        self._Z = shrinkge_fn(B, self.theta, 'Z_0')
        tf.add_to_collection('SC_Zt', self._Z)
        #
        # run unrolling
        for self.t in range(1, self._unroll_count):

            conv_wd = self._conv2d_dec(
                _val=self._Z,
                _name='convWd'
                )
            res = self._Z - conv_wd
            res_add_bias = res + B
            self._Z = shrinkge_fn(
                res_add_bias,
                self.theta,
                'Z_'+str(self.t)
                )
            tf.add_to_collection('SC_Zt', self._Z)

