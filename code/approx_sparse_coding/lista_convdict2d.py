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
        def __init__(
            self, unroll_count,
            inputshape=None,
            filter_arr=None,
            batch_size=None,
            channel_size=3,
            shrinkge_type='soft_thresh',
            **kwargs
            ):

        """  
        """
        super(LISTAConvDict2dBase, self).__init__()


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