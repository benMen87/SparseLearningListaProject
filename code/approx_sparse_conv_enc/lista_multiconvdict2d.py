"""
This module contanint Class LISTAMultipleConvDict2d
"""
import sys
import tensorflow as tf
import numpy as np
import lista_convdict2d_base


class LISTAConvMultiDict2d(lista_convdict2d_base.LISTAConvDict2dBase):
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
            shrinkge_type='soft_thresh',
            **kwargs
            ):

        kwargs['inputshape'] = inputshape
        kwargs['batch_size'] = batch_size
        kwargs['channel_size'] = channel_size
        kwargs['shrinkge_type'] = shrinkge_type

        super(LISTAConvMultiDict2d, self).__init__(
            unroll_count=unroll_count, **kwargs)
        self.expand_amount = kwargs.get('expand_amount', 1)

    def _conv2d_enc(self, _val, _name='enc'):
        res = self.conv2d(_val, self._We_expand, _name)
        return res

    def _conv2d_dec(self, _val, _name):
        res = self.conv2d(_val, self._Wd_expand, _name)
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
        super(LISTAConvMultiDict2d, self).init_random_ista_coherent(kwargs)
        self.expand_amount = kwargs.get('expand_amount', 3)

        we_list = tf.split(self._We, num_or_size_splits=self.amount_of_kernals,
                axis=3)
        for _ in range(self.expand_amount):
            we_list = self.expand_perseptive_field(we_list)
        we_list = 0.01 * tf.nn.l2_normalize(we_list, dim=[0,1])
        self._We_expand = tf.stack(we_list, axis=3)
        if self._norm_kers:
            self._We_expand = 0.01 * tf.nn.l2_normalize(self._We_expand, dim=[0,1])

        wd_list = tf.split(self._Wd, num_or_size_splits=self.amount_of_kernals,
                axis=2)
        for _ in range(self.expand_amount):
            wd_list = self.expand_perseptive_field(wd_list)
        wd_list = tf.nn.l2_normalize(wd_list, dim=[0,1])

        self._We_expand = tf.stack(wd_list, axis=2)
        if self._norm_kers:
            self._Wd_expand = tf.nn.l2_normalize(self._Wd_expand, dim=[0,1])

        print('We {}'.format(self._We_expand.get_shape()))
        print('Wd {}'.format(self._Wd_expand.get_shape()))

    def expand_perseptive_field(self, filter_list):
        """expand reseptive feild of filter.

       Expand reseptive field for each filter in filter_list,
       by convolving 'FULL' each filter with itself with a reflective padding. 

        Arguments:
            filter_list {list} -- list of filter.

        Returns:
            filter_list_expand {list} -- list of expanded resptive feild filters.
        """
        filter_list_expand = []
        pad_sz = self.kernel_size // 2
        for f in filter_list:
            if len(f.get_shape().as_list()) == 4:
                f = tf.squeeze(f, axis=3)
            pad_f = tf.expand_dims(tf.pad(f, [[pad_sz, pad_sz], [pad_sz, pad_sz], [0, 0]], "REFLECT" ), 0)
            f = tf.expand_dims(f, axis=-1)
            filter_list_expand.append(tf.squeeze(tf.nn.conv2d(pad_f, f,
                strides=[1,1,1,1], padding='SAME'), axis=0))
        return filter_list_expand

