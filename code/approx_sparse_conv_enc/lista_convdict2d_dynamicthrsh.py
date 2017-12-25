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

class LISTARandNoiseDynamicThrsh(object):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """
    def __init__(
            self,
            thrsh_scale_factor,
            sigmas,
            sigma_scale_factor,
            is_train=True
            ):

        self.sigmas = sigmas
        self.thrsh_scale_factor = thrsh_scale_factor
        self.sigma_scale_factor = sigma_scale_factor
        self.noise_sigma = 0 if is_train else tf.placeholder_with_default(0, shape=())

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise

    def noise_adaptive_threshold(self, inputs):
        #TODO:  better way to implement add noise vs givin noise? 
        """
        If noise_sigma placeholder is set then input is noisy 
        if noise_sigma is None then sample noise
        """

        if self.noise_sigma == 0: #i.e. input clean we sample and add noise
            smpl_sigma = tf.random_shuffle(self.sigmas)
            smpl_sigma = smpl_sigma[0]
            self._inputs_noisy = self.gaussian_noise_layer(inputs, tf.to_float(smpl_sigma)/self.sigma_scale_factor) # add noise
        else:
            self._inputs_noisy = inputs 
            smpl_sigma = self.noise_sigma
        tf.summary.scalar('smpl_sigma', smpl_sigma)
        self._scale_thrsh = tf.to_float(smpl_sigma) / self.thrsh_scale_factor

    @property
    def inputs_noisy(self):
        return self._inputs_noisy

class LISTAConvDict2dDynamicThrsh(lista_convdict2d.LISTAConvDict2d):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """
    def __init__(
            self,
            thrsh_scale_factor=20.0,
            sigmas=tf.range(5, 51, 5),
            sigma_scale_factor=255,
            **kwargs
            ):

        super(LISTAConvDict2dDynamicThrsh, self).__init__(
            **kwargs
            )

        self._dynamic_noise_layer = LISTARandNoiseDynamicThrsh(
            thrsh_scale_factor=thrsh_scale_factor,
            sigmas=sigmas,
            sigma_scale_factor=sigma_scale_factor)

    def build_model(self, inputs):
        self._dynamic_noise_layer.noise_adaptive_threshold(inputs)
        super(LISTAConvDict2dDynamicThrsh, self).build_model(self.inputs_noisy)

    @property
    def inputs_noisy(self):
        return self._dynamic_noise_layer._inputs_noisy

    @property
    def theta(self):
        _theta = tf.clip_by_value(self._theta, 0, 1)
        return self._dynamic_noise_layer._scale_thrsh * _theta

class LISTAConvDict2dDynamicThrshUntied(lista_convdict2d_untied.LISTAConvDict2dUntied):
    """Class of approximate SC based on 2D convolutinal dictioary.
       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
    """
    def __init__(
            self,
            thrsh_scale_factor=20.0,
            sigmas=tf.range(5, 51, 5),
            sigma_scale_factor=255,
            **kwargs
            ):

        super(LISTAConvDict2dDynamicThrshUntied, self).__init__(
            **kwargs
            )

        self._dynamic_noise_layer = LISTARandNoiseDynamicThrsh(
            thrsh_scale_factor=thrsh_scale_factor,
            sigmas=sigmas,
            sigma_scale_factor=sigma_scale_factor)

    def build_model(self, inputs):
        self._dynamic_noise_layer.noise_adaptive_threshold(inputs)
        super(LISTAConvDict2dDynamicThrshUntied, self).build_model(self.inputs_noisy)

    @property
    def inputs_noisy(self):
        return self._dynamic_noise_layer._inputs_noisy

    @property
    def theta(self):
        _theta = super(LISTAConvDict2dDynamicThrshUntied, self).theta
        _theta = tf.clip_by_value(_theta, 0, 1)
        return self._dynamic_noise_layer._scale_thrsh * _theta

#class LISTAConvDict2dDynamicThrsh(lista_convdict2d.LISTAConvDict2d):
#    """Class of approximate SC based on 2D convolutinal dictioary.
#       x = sum(f_l*Z_l) = (circ(f_0)|circ(f_1)|...|circ(f_n))[Z_0|..|Z_n]
#       -> Wd = (circ(f_0)|circ(f_1)|...|circ(f_n)) or 1 filter with depth of n
#       -> We = (circ(f_0)|circ(f_1)|...|circ(f_n))^T or n fiters with depth of 1
#    """
#    def __init__(
#            self,
#            thrsh_scale_factor=20.0,
#            sigmas=[5,10,15,20,25,30],
#            sigma_scale_factor=255,
#            **kwargs
#            ):
#
#        super(LISTAConvDict2dDynamicThrsh, self).__init__(
#            **kwargs
#            )
#
#        self.sigmas = sigmas
#        self.thrsh_scale_factor = thrsh_scale_factor
#        self.sigma_scale_factor = sigma_scale_factor
#        self.is_train = tf.placeholder_with_default(True,shape=(), name='is_train')
#        self.scale_thrsh = 0
#        self._inputs_noisy = None
#
#    def gaussian_noise_layer(self, input_layer, std):
#        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
#        return input_layer + noise
#
#    
#    def build_model(self, inputs):
#        #TODO: add cond if training
#        smpl_sigma = tf.random_shuffle(self.sigmas)
#        smpl_sigma = smpl_sigma[0]
#        ## tf.py_func(
#        #    np.random.choice,
#        #    [self.sigmas,
#        #    1,
#        #    False,
#        #    [[1./len(self.sigmas)]*len(self.sigmas)]],
#        #    tf.int32
#        #    )
#        self._inputs_noisy = self.gaussian_noise_layer(inputs, tf.to_float(smpl_sigma)/self.sigma_scale_factor)
#        self.scale_thrsh = tf.to_float(smpl_sigma) / self.thrsh_scale_factor
#
#        super(LISTAConvDict2dDynamicThrsh, self).build_model(self._inputs_noisy)
#
#    @property
#    def inputs_noisy(self):
#        return self._inputs_noisy
#
#    @property
#    def theta(self):
#        _theta = tf.clip_by_value(self._theta, 0, 1)
#        return self.scale_thrsh * _theta