"""
This module contanint Class LISTAMultipleConvDict2d
"""
import sys
import tensorflow as tf
import numpy as np
import lista_convdict2d_base


class LISTAConvDict2dUntied(lista_convdict2d_base.LISTAConvDict2dBase):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """
    def __init__(
            self, unroll_count,
            inputshape=None,
            batch_size=None,
            channel_size=3,
            **kwargs
            ):
        print kwargs
        kwargs['inputshape'] = inputshape
        kwargs['batch_size'] = batch_size
        kwargs['channel_size'] = channel_size

        self.curr_layer = 0
        self._We = []
        self._Wd = []
        self._theta = []

        super(LISTAConvDict2dUntied, self).__init__(
            unroll_count=unroll_count, **kwargs)

    def _conv2d_enc(self, _val, _name='enc'):
        if self._norm_kers:
            _Wet = tf.nn.l2_normalize(self.We, dim=[0,1])
        else:
            _Wet = self.We
        res = self.conv2d(_val, _Wet, _name)
        return res

    def _conv2d_dec(self, _val, _name):
        if self._norm_kers:
            _Wdt = tf.nn.l2_normalize(self.Wd, dim=[0,1])
        else:
            _Wdt = self.Wd

        res = self.conv2d(_val, _Wdt, _name)
        return res

    def conv2d(self, _value, _filters, _name, _dilate=False, _rate=[1, 2, 3]):
        res = tf.nn.conv2d(
                _value,
                _filters,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name=_name
            )
        return res

    def init_random_ista_coherent(self, kwargs):
        self.amount_of_kernals = kwargs.get('kernel_count', 64)
        self.kernel_size = kwargs.get('kernel_size', 7)

        thrsh = kwargs.get('init_threshold', 1e-8)
        init_We = tf.nn.l2_normalize(tf.truncated_normal([self.kernel_size, self.kernel_size,
                                      self.input_channels, self.amount_of_kernals]), dim=[0,1])
        init_Wd = tf.transpose(tf.reverse(init_We, [0,1]), [0,1,3,2])
        for _ in range(self._unroll_count):
            self._We.append(tf.Variable(0.1 * init_We, name='We'))
            self._Wd.append(tf.Variable(init_Wd, name='Wd'))
            self._theta.append(tf.nn.relu(tf.Variable(tf.fill([1, self.amount_of_kernals], thrsh), name='theta')))


    @property
    def Wd(self):
        return self._Wd[self.t]

    @property
    def We(self):
        return  self._We[self.t]

    @property
    def theta(self):
        return self._theta[self.t]
     
