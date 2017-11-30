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


    def _conv2d_enc(self, _val, _name='enc'):
        _We = [tf.nn.l2_normalize(0.001 * we, [0,1]) for we in self._We ]
        res = _val
        for we in _We:
            res = self.conv2d(res, we, _name)
        return res

    def _conv2d_dec(self, _val, _name):
        _Wd = [tf.nn.l2_normalize(wd, [0,1]) for wd in self._Wd]
        res = _val
        for wd in _Wd:
            res = self.conv2d(res, wd, _name) 
        return res
     
    def conv2d(self, _value, _filters, _name, _dilate=False, _rate=[1, 2, 3]):
        if _dilate:
            res = dilate_conv2d(_value, _filters, _name, 3)
        else:
            res = tf.nn.conv2d(
                _value,
                _filters,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name=_name
            )
        return res
 
    def init_random_ista_coherent(self, kwargs):
        """
        Override base impl of init for multiple dicts
        """
        self.amount_of_kernals = [32, 64, 64]
        ker_shapes = [
            (self.input_channels, self.amount_of_kernals[0]),
            (self.amount_of_kernals[0], self.amount_of_kernals[1]),
            (self.amount_of_kernals[1], self.amount_of_kernals[1])]
        self.kernel_size = kwargs.get('kernel_size', 3)

        if self._shrinkge_type == 'soft thresh':
            #TODO: Notice thresh is now shared one for each feture map
            thrsh = kwargs.get('init_threshold', 0.1)
            self._theta = tf.nn.relu(tf.Variable(tf.fill([1,
                self.amount_of_kernals[-1]],
                thrsh), name='theta'))
          #       self.amount_of_kernals], value=0.01)), name='theta')] * unroll_count
        elif self._shrinkge_type == 'smooth soft thresh':
            beta = [tf.Variable(tf.fill([1, self.amount_of_kernals], 5.0), name='beta'+str(u))
                          for u in range(unroll_count)]
            b = [tf.Variable(tf.fill([1, self.amount_of_kernals], 0.5), name='b'+str(u))
                       for u in range(unroll_count)]
            self._theta = zip(beta, b)
        init_We = [tf.nn.l2_normalize(tf.truncated_normal([self.kernel_size,
            self.kernel_size, d0, d1]), dim=[0,1]) for d0, d1 in ker_shapes ]
        self._We = [tf.Variable(_init_We, name='We') for _init_We in init_We]
        self._Wd = [tf.Variable(tf.transpose(tf.reverse(we.initialized_value(),
            [0,1]), [0,1,3,2]), name='Wd') for we in reversed(self._We)]

