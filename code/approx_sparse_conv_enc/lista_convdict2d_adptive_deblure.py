"""
This module contanint Class LISTAConvDict2d
That an TF model for approximation CSC.
The model will scale  its thrshold based on noise sigma.
"""
import sys
import tensorflow as tf
import numpy as np
import lista_convdict2d
import lista_convdict2d_untied

class TFPSFLayer(object):
    """    """
    def __init__(self, sigma=8):

        print('is train: {}'.format(is_train))
        self._psfs = tf.stack([self._creat_psf1(), self._creat_psf2(), self._creat_psf3()])
        self._sigma = float(sigma) / 255

    def _creat_psf1(self):
        ker = np.empty(shape=(15, 15))
        for x1 in range(-7, 8):
            for x2 in range(-7, 8):
                ker[x1+7, x2+7] = 1.0 / (1 + x1**2 + x2**2)
        ker /= np.sum(ker)
        return tf.Variable(ker[...,None], trainable=False)

    def _creat_psf2(self):
        ker = np.ones(shape=(9, 9, 1))
        ker /= np.sum(ker)
        return tf.Variable(ker[...,None], trainable=False)

    def _creat_psf3(self):
        row = np.array([1, 4, 6, 4, 1])
        ker = row.T * row
        ker /= np.sum(ker)
        return tf.Variable(ker[...,None], trainable=False)

    def __call__(self, inputs):
        """
        Train-time function. choose random psf from list and add noise.
        """

        _psf = tf.random_shuffle(self._psfs)[0]
        inputs_blur = tf.nn.conv2d(inputs, _psf, strides[1,1,1,1], padding='SAME')
        inputs_blur += tf.random_normal(inputs.shape, self._sigma)
        return inputs_blur, _psf

class LISTAConvDict2dAdaptiveBlur(lista_convdict2d.LISTAConvDict2d):
    """One model multiple blur kernels"""
    def __init__(
            self,
            **kwargs
            ):

        super(LISTAConvDict2dAdaptiveBlur, self).__init__(
            **kwargs
            )
        # TODO: add test support select specific psf
        self.is_train = kwargs.get('is_train', True)
        self._blur_layer = TFPSFLayer()
        self._mask = None
        self._inputs_blur = None

    def build_model(self, inputs):
        self._inputs_blur, self._psf = self._blur_layer(inputs)
        super(LISTAConvDict2dAdaptiveBlur, self).build_model(self.inputs_noisy)

    def _creat_mask(self, shape):
        # Overide base class _creat_mask method.
        self._mask = self._psf
    
    def _apply_mask(self, inputs):
        return tf.nn.conv2d(inputs,
                            self._mask,
                            strides=[1,1,1,1],
                            padding='SAME'
                            )

    @property
    def inputs_blur(self):
        return self._inputs_blur

class LISTAConvDict2dAdaptiveBlurUntied(lista_convdict2d_untied.LISTAConvDict2dUntied):
    """One model multiple blur kernels"""
    def __init__(
            self,
            **kwargs
            ):

        super(LISTAConvDict2dAdaptiveBlur, self).__init__(
            **kwargs
            )
        # TODO: add test support select specific psf
        self.is_train = kwargs.get('is_train', True)
        self._blur_layer = TFPSFLayer()
        self._mask = None
        self._inputs_blur = None

    def build_model(self, inputs):
        self._inputs_blur, self._psf = self._blur_layer(inputs)
        super(LISTAConvDict2dAdaptiveBlur, self).build_model(self.inputs_noisy)

    def _creat_mask(self, shape):
        # Overide base class _creat_mask method.
        self._mask = self._psf
    
    def _apply_mask(self, inputs):
        return tf.nn.conv2d(inputs,
                            self._mask,
                            strides=[1,1,1,1],
                            padding='SAME'
                            )

    @property
    def inputs_blur(self):
        return self._inputs_blur
