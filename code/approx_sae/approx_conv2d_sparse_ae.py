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
from approx_sparse_conv_enc import lista_multiconvdict2d
from approx_sparse_conv_enc import lista_convdict2d_untied
from approx_sparse_conv_enc import lista_convdict2d_dynamicthrsh
from approx_sparse_conv_enc import lista_convdict2d_adptive_deblure
from approx_sparse_conv_dec import dec_convdict2d

class AutoEncoderBase(object):
    __metaclass__ = ABCMeta

    class NotInitlizedError(Exception):
        pass

    def __init__(self):
        self._encoder = []
        self._decoder = []

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
    LEGALE_TPYES = [
        'convdict',
        'convmultidict',
        'untied',
        'dynamicthrsh',
        'dynamicthrsh_untied',
        'adaptive_deblur',
        'adaptive_deblur_untied'
    ]

    def __init__(self, type='convdict', **kwargs):
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
        return self.raise_on_none(self._outputs, 'output')

    @property
    def sparsecode(self):
        return self.raise_on_none(self.encoder.output, 'output')

    @property
    def input(self):
        return self.raise_on_none(self.inputs, 'intput')

    @property
    def encoder(self):
        return self.raise_on_none(self._encoder[-1], 'LISTAencoder')
    
    @property
    def decoder(self):
        return self.raise_on_none(self._decoder[-1], 'CSCdecoder')

    @property
    def type(self):
        return self._type

    # CREATING MODEL
    def _get_encoder(self, **encargs):
        if self._type == 'convdict':
            encoder_fn = lista_convdict2d.LISTAConvDict2d
        elif self._type == 'convmultidict':
            encoder_fn = lista_multiconvdict2d.LISTAConvMultiDict2d
        elif self._type == 'untied':
            encoder_fn = lista_convdict2d_untied.LISTAConvDict2dUntied
        elif self._type == 'dynamicthrsh':
            encoder_fn = lista_convdict2d_dynamicthrsh.LISTAConvDict2dDynamicThrsh
        elif self._type == 'dynamicthrsh_untied':
            encoder_fn = lista_convdict2d_dynamicthrsh.LISTAConvDict2dDynamicThrshUntied
        elif self._type == 'adaptive_deblur':
            encoder_fn = lista_convdict2d_adptive_deblure.LISTAConvDict2dAdaptiveBlur
        elif self._type == 'adaptive_deblur_untied':
            encoder_fn = lista_convdict2d_adptive_deblure.LISTAConvDict2dAdaptiveBlurUntied
        else:
            # shouldn't reach here
            raise SystemError("Shouldn't reach here")
        with tf.variable_scope('encoder'):
            encoder = encoder_fn(**encargs)
        return encoder

    def _get_decoder(self, **decargs):
        if self._type == 'convmultidict':
            decoder_fn = dec_convdict2d.DecConvMultiDict2d
        else:
            decoder_fn = dec_convdict2d.DecConvDict2d

        with tf.variable_scope('decoder'):
            decoder = decoder_fn(
               init_val=decargs['init_val'],
               output_shape=decargs['output_size'],
               norm_kernal=decargs['norm_kernal']
            )
        return decoder

    def _creat_input_placeholder(self, w=None, h=None, c=None):
        inpt = tf.placeholder(tf.float32,
        shape=(None, w, h, c),
        name='X')
        return inpt

    def _add_ae_block(self, inputs, encargs):
        self._encoder.append(self._get_encoder(**encargs))
        self._encoder[-1].build_model(inputs)
        _decargs = {
            'init_val':self._encoder[-1].Wd, # TODO: Fix this.initialized_value(),
            'output_size':self._encoder[-1].inputshape,
            'norm_kernal': self._encoder[-1]._norm_kers
        }
        self._decoder.append(self._get_decoder(**_decargs))
        self._decoder[-1].build_model(_sc=self._encoder[-1].output)
        return self._decoder[-1].output

    def _downsample_layer(self, inputs, s=2):
        downsmapled = tf.layers.conv2d(inputs=inputs, filters=inputs.get_shape().as_list()[-1], kernel_size=(5, 5), strides=(s, s), use_bias=False)
        return downsmapled

    def _upsample_layer(self, inputs, s=2):
        upsmapled = tf.layers.conv2d_transpose(inputs=inputs, filters=inputs.get_shape().as_list()[-1], kernel_size=(5, 5), strides=(s, s), use_bias=False)
        return upsmapled

    def register_tb_images(self, tf_summary):
        if 'dynamicthrsh' in  self.type:
            tf_summary.image('input', self.encoder.inputs_noisy)
        elif 'adaptive_deblur' in self.type:
            tf_summary.image('input', self.encoder.inputs_blur)
        else:
            tf_summary.image('input', self.input)

        if 'residual' in self.sae_type:
            tf_summary.image('downsampled_input', self._downsample_inputs)
            tf_summary.image('downsampled_out', self._downsampled_out) 
            tf_summary.image('residual_in', self._inputs_residual)
            tf_summary.image('residual_out', self._outputs_residual)
        tf_summary.image('output', _model.output)
        tf_summary.image('target', _model.target)

    def build_model(self, sae_type='classic_sae', **encargs):
        """
        encoder-args can be user choice.
        decoder-args is defined from encoder-args.
        """
        self.inputs = self._creat_input_placeholder(c=encargs['channel_size'])

        if sae_type == 'classsic_sae':
            self.sae_type = 'classic'
            output = self._add_ae_block(self.inputs, encargs)
            self._outputs = tf.clip_by_value(output, 0, 1)
        elif sae_type == 'residual_sae':
            self.sae_type = 'residual'
            self._downsample_inputs = self._downsample_layer(self.inputs)
            self._downsampled_out = self._add_ae_block(self._downsample_inputs, encargs) # downsample and clean
            self._upsampled_out = self._upsample_layer(self._downsampled_out)
            self._upsampled_out = tf.clip_by_value(self._upsampled_out, 0, 1)

            self._inputs_residual = self.input - self._upsampled_out
            self._outputs_residual = self._add_ae_block(self._inputs_residual, encargs)
            self._outputs = tf.clip_by_value(self._outputs_residual, 0, 1) + self._upsampled_out
 