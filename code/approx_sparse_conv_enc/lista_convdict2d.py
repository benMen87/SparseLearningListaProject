"""
This module contanint Class LISTAConvDict2d
That an TF model for approximation CSC
"""
import sys
import tensorflow as tf
import numpy as np
import lista_convdict2d_base


class LISTAConvDict2d(lista_convdict2d_base.LISTAConvDict2dBase):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """
    def __init__(
            self, unroll_count,
            inputshape=None,
            batch_size=None,
            input_channels=1,
            shrinkge_type='soft_thresh',
            **kwargs
            ):

        kwargs['inputshape'] = inputshape
        kwargs['batch_size'] = batch_size
        kwargs['input_channels'] = input_channels
        kwargs['shrinkge_type'] = shrinkge_type

        super(LISTAConvDict2d, self).__init__(
            unroll_count=unroll_count, **kwargs
            )

    def _conv2d(self, _val, _filters, _name):
        res = tf.nn.conv2d(
            _value,
            _filters,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name=_name
        )
        return res

    def _conv2d_enc(self, _val, _name='enc'):
        _We = 0.1 * tf.nn.l2_normalize(self._We, [0,1])
        return self.conv2d(_val, _We, _name) 

    def _conv2d_dec(self, _val, _name):
        _Wd = tf.nn.l2_normalize(self._Wd, [0,1])
        return self.conv2d(_val, _Wd, _name) 
     
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
