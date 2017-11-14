"""
This model is an approximate convoulution sparse model.
Implementation of paper: https://arxiv.org/pdf/1711.00328.pdf
"""


import os
import sys
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from approx_sparse_conv_enc import lista_convdict2d
from approx_sparse_conv_dec import dec_convdict2d

class AutoEncoderBase(object):
    __metaclass__ = ABCMeta

    class NotInitlizedError(Exception):
        pass

    def __init__(self):
        self._encoder = None
        self._decoder = None

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def output(self):
        pass

    @property
    @abstractmethod
    def input(self):
        pass

    def raise_on_none(self, _val, _name):
        if _val is None:
            raise NotInitlizedError('variable {} is not inilized. (run "build_model" first)')
        else:
            return _val
   
    
    
class ApproxCSC(AutoEncoderBase):
    LEGALE_TPYES = ['classic']

    def __init__(self, type='classic', **kwargs):
        super(ApproxCSC, self).__init__()

        if type not in self.LEGALE_TPYES:
            raise TypeError('Bad requested ACSC type. choose type from \
            list {}'.format(self.LEGALE_TPYES))
        self._type = type

    #  GETTERS  - I/O
    @property
    def target(self):
        return self.raise_on_none(self.decoder.target, 'target')

    @property
    def output(self):
        return self.raise_on_none(self.decoder.output, 'output')

    @property
    def sparsecode(self):
        return self.raise_on_none(self.encoder.output, 'output')

    @property
    def input(self):
        return self.raise_on_none(self.encoder.input, 'intput')

    @property
    def encoder(self):
        return self.raise_on_none(self._encoder, 'LISTAencoder')
    
    @property
    def decoder(self):
        return self.raise_on_none(self._decoder, 'CSCdecoder')

    # CREATING MODEL
    def _get_encoder(self, **encargs):
        if self._type == 'classic':
            encoder_fn = lista_convdict2d.LISTAConvDict2d
        else:
            # shouldn't reach here
            raise SystemError("Shouldn't reach here")
        with tf.variable_scope('encoder'):
            encoder = encoder_fn(**encargs)
        return encoder

    def _get_decoder(self, **decargs):
        if self._type == 'classic':
            decoder_fn = dec_convdict2d.DecConvDict2d
        else:
            # shouldnt reach here
            raise SystemError("Shouldn't reach here")

        with tf.variable_scope('decoder'):
           decoder = decoder_fn(
               decargs['init_val'],
               decargs['output_size']
            )
        return decoder

    def build_model(self, **encargs):
        """
        encoder-args can be user choice.
        decoder-args is defined from encoder-args.
        """
        self._encoder = self._get_encoder(**encargs)
        self._encoder.build_model()

        _decargs = {
            'init_val':self._encoder.Wd.initialized_value(),
            'output_size':self._encoder.inputshape
        }
        self._decoder = self._get_decoder(**_decargs)
        self._decoder.build_model(_sc=self._encoder.output)
