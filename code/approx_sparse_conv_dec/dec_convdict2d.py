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
    def __init__(self, init_val, output_shape, norm_kernal):
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
    def __init__(self, init_val, output_shape, norm_kernal):
        super(DecConvMultiDict2d, self).__init__(init_val, output_shape)
    
    def _init_cd(self, init_val):
        #TODO: make num_of_splits and expand_count dynamic
        num_of_splits = init_val.get_shape().as_list()[2]
        expaned_count = 1
        self._cd = tf.Variable(init_val.initialized_value(), name='decoder')
        cd_list = tf.split(self._cd, num_or_size_splits=num_of_splits, axis=2)
        pad_sz =  cd_list[0].get_shape().as_list()[0] // 2 
        for _ in range(expaned_count):
            cd_list = self.expand_perseptive_field(cd_list, pad_sz)
        _cd = tf.nn.l2_normalize(tf.stack(cd_list, axis=2), dim=[0,1])
        self._cd = tf.nn.l2_normalize(_cd, dim=[0, 1], name='normilized_dict')
        print(self._cd.get_shape().as_list())

    def reconstruct(self, _sc):
        res =  tf.nn.conv2d(
            _sc,
            self._cd,
            strides=[1, 1, 1, 1],
            padding='SAME'
            )
        return res

    def expand_perseptive_field(self, filter_list, pad_sz):
        """expand reseptive feild of filter.

       Expand reseptive field for each filter in filter_list,
       by convolving 'FULL' each filter with itself with a reflective padding. 

        Arguments:
            filter_list {list} -- list of filter.

        Returns:
            filter_list_expand {list} -- list of expanded resptive feild filters.
        """
        filter_list_expand = []
        for f in filter_list:
            if len(f.get_shape().as_list()) == 4:
                f = tf.squeeze(f, axis=3)
            pad_f = tf.expand_dims(tf.pad(f, [[pad_sz, pad_sz], [pad_sz, pad_sz], [0, 0]], "REFLECT" ), 0)
            f = tf.expand_dims(f, axis=-1)
            filter_list_expand.append(tf.squeeze(tf.nn.conv2d(pad_f, f,
                strides=[1,1,1,1], padding='SAME'), axis=0))
        return filter_list_expand

