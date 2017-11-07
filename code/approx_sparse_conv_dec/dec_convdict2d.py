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
        super(DecConvDict2d, self).__init__(init_val, output_shape)

    def reconstruct(self, _sc):
        res =  tf.nn.conv2d(
            _sc,
            self._cd,
            strides=[1, 1, 1, 1],
            padding='SAME'
            )
        return res
