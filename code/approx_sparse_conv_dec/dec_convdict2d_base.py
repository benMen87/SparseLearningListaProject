"""
 This module contains Class DecConvDict2dBase,
 Is the base class for approximate convolution sparse coding convolutional dictionary.
"""


from abc import ABCMeta, abstractmethod
import sys
import tensorflow as tf
import numpy as np



class DecConvDict2dBase(object):
    
    __metaclass__ = ABCMeta

    def __init__(self, init_val, output_size):
        self._cd = tf.Variable(init_val, name='decoder') 
        self._cd = tf.nn.l2_normalize(self._cd, dim=[0, 1], name='normilized_dict') # keep decoder atoms with l2 norm of 1
        self._target = tf.placeholder(tf.float32, shape=output_size)
        self._output = None

    @abstractmethod
    def reconstruct(self, _sc):
        """
        return should be somthing of the sort of:
            ret = conv2d(sc, dict)
        """

    def build_model(self, _sc):
        """Can pass multiple inputs at diffrent time state of encoder"""
        self._output = self.reconstruct(_sc)

    @property
    def output(self):
        return self._output

    @property
    def target(self):
        return self._target

    @property
    def convdict(self):
        return self._cd
    

    

