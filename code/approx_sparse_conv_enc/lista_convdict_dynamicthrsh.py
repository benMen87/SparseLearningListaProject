"""
This module contanint Class LISTAConvDict2d
That an TF model for approximation CSC.
The model will scale  its thrshold based on noise sigma.
"""
import sys
import tensorflow as tf
import numpy as np
import lista_convdict2d


class LISTAConvDict2dDynamicThrsh(lista_convdict2d.LISTAConvDict2d):
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
            **kwargs
            ):

        super(LISTAConvDict2dDynamicThrsh, self).__init__(
            **kwargs
            )

        self.sigmas = sigmas
        self.thrsh_scale_factor = thrsh_scale_factor
        self.sigma_scale_factor = sigma_scale_factor
        self.is_train = tf.placeholder_with_default(False, shap=(,), name='is_train')
        self.scale_thrsh = 0

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise

    
    def build_model(self, inputs):
        smpl_sigma = tf.py_func(
            np.random.choice,
            self.sigmas,
            1,
            False,
            [[1./len(sigmas)]*len(sigmas]),
            tf.int32
            )
        inputs_noisy = self.gaussian_noise_layer(inputs, float(smpl_sigma)/self.sigma_scale_factor)
        scale_thrsh = float(smpl_sigma) / self.thrsh_scale_factor

        super(LISTAConvDict2dDynamicThrsh, self).build_model(inputs_noisy)


    @property
    def theta(self):
        return self.scale_thrsh * self._theta




