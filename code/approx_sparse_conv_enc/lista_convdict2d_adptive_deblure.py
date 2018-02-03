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
    def __init__(self, psf_id=-1):

        self._fn_psfs =  [self._creat_psf1, self._creat_psf2, self._creat_psf3]
        self._sigma = tf.sqrt(2.) / 255  # sigma^2 = 2
        self._psf_id = psf_id

    def _creat_psf1(self):
        ker = np.empty(shape=(15, 15)).astype('float32')
        for x1 in range(-7, 8):
            for x2 in range(-7, 8):
                ker[x1+7, x2+7] = 1.0 / (1 + x1**2 + x2**2)
        ker /= np.sum(ker)
        ker = ker[..., None]
        return tf.constant(ker) 

    def _creat_psf2(self):
        ker = np.ones(shape=(9, 9)).astype('float32');
        ker /= np.sum(ker)
        ker = ker[..., None]
        return tf.constant(ker)

    def _creat_psf3(self):
        row = np.array([[1, 4, 6, 4, 1]]).astype('float32')
#        ker = np.pad(row.T * row, 5, 'constant', constant_values=0) # padd 0
        ker = row.T * row
        ker /= np.sum(ker)
        ker = ker[..., None]
        return tf.constant(ker)

    def _get_psf(self):
        """
        Linear combiantion of of the 3 defined psfs
        """
        if self._psf_id == -1:
            alphas = tf.random_uniform(shape=[3])
            alphas /= tf.reduce_sum(alphas)
            _scale = tf.random_uniform(shape=[1], minval=5,  maxval=15, dtype=tf.int32)
            _psfs = [ tf.image.resize_images(fn_psf(), size=[_scale[0],
                _scale[0]])[...,None] for fn_psf in self._fn_psfs ]
            self._psfs = tf.concat(_psfs, axis=-1)
            # Random linear combination of filters
            _psf = tf.reduce_sum(alphas * self._psfs, axis=-1, keep_dims=True)
            # Reduce sum to 1.
            _psf /= tf.reduce_sum(_psf) 

        else:
            _psf = self._fn_psfs[self._psf_id]()[...,None]
        return _psf


    def __call__(self, inputs):
        """
        Train-time function. choose random psf from list and add noise.
        """

        _psf = self._get_psf()
        inputs_blur = tf.nn.conv2d(inputs, _psf, strides=[1,1,1,1], padding='SAME')
        inputs_blur += tf.random_normal(shape=tf.shape(inputs),
                stddev=self._sigma, dtype=tf.float32) 
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
        self._psf_id = kwargs.get('psf_id')
        self._blur_layer = TFPSFLayer(self._psf_id)
        self._mask = None
        self._inputs_blur = None

    def build_model(self, inputs):
        self._inputs_blur, self._psf = self._blur_layer(inputs)
        super(LISTAConvDict2dAdaptiveBlur, self).build_model(self.inputs_blur)

    def _creat_mask(self, shape):
        # Overide base class _creat_mask method.
        self._mask = self._psf
    
    def _apply_mask(self, inputs):
        apply_mask = tf.nn.conv2d(inputs, self._mask, strides=[1,1,1,1], padding='SAME')
        return apply_mask_t

    @property
    def inputs_blur(self):
        return self._inputs_blur

class LISTAConvDict2dAdaptiveBlurUntied(lista_convdict2d_untied.LISTAConvDict2dUntied):
    """One model multiple blur kernels"""
    def __init__(
            self,
            **kwargs
            ):

        super(LISTAConvDict2dAdaptiveBlurUntied, self).__init__(
            **kwargs
            )
        # TODO: add test support select specific psf
        self.is_train = kwargs.get('is_train', True)
        self._psf_id = kwargs.get('psf_id')
        self._blur_layer = TFPSFLayer(self._psf_id)

        self._mask = None
        self._inputs_blur = None

    def build_model(self, inputs):
        self._inputs_blur, self._psf = self._blur_layer(inputs)
        super(LISTAConvDict2dAdaptiveBlurUntied, self).build_model(self._inputs_blur)

    def _creat_mask(self, shape):
        # Overide base class _creat_mask method.
        self._mask = self._psf
    
    def _apply_mask(self, inputs):
        return tf.nn.conv2d(inputs, self._mask, strides=[1,1,1,1],
                padding='SAME')

    @property
    def inputs_blur(self):
        return self._inputs_blur
