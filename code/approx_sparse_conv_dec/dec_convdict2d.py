"""
This module is an implemintation of a sparse convolution dictionay.
"""


import dec_convdict2d_base
import sys
import tensorflow as tf
import numpy as np
from  dec_convdict2d_base import DecConvDict2dBase

class DecConvDict2d(DecConvDict2dBase):

    """
    Decoder is a simple linear convolutoin dictionay
    """
    def __init__(self, init_val, output_shape):
        super(DecConvDict2d, self).__init__(init_val.initialized_value(), output_shape)

    def reconstruct(self, _sc):
        res =  tf.nn.conv2d(
            _sc,
            self._cd,
            strides=[1, 1, 1, 1],
            padding='SAME'
            )
        return res


class DecConvMultiDict2d(DecConvDict2dBase):

    """
    Decoder is a simple linear convolutoin dictionay
    """
    def __init__(self, init_val, output_shape):
        super(DecConvMultiDict2d, self).__init__(init_val, output_shape)
    
    def _init_cd(self, init_val):
        self._cd = tf.Variable(init_val, name='decoder')
        self._cd = [tf.nn.l2_normalize(self._cd, dim=[0, 1], name='normilized_dict') 
