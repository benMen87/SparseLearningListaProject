from lcod import LCoD
import tensorflow as tf


class Lmnist(object):
    """Testing MNIST classification by passing full 28X28 through LCOD and then 
       into simple FC layer and then soft max all should learn to gather.
    """
    NUMBER_OF_CALSSES = 10
    def __init__(self, We_shape, unroll_count, number_of_hidden=200, We=None):

        self._locd              = LCoD(We_shape, unroll_count, We)
        #
        # pass full size
        m, n                    = We_shape
        self._input             = self._locd.output
        self._labels            = tf.placeholder(tf.float32, shape=(Lmnist.NUMBER_OF_CALSSES, 1))
        self.number_of_hidden   = number_of_hidden
        self._pred              = None
        self._logits            = None
        self._classify_loss     = None
        #
        # model vars
        #
        # FC in
        h_in_shape              = (self.number_of_hidden, m)
        self.h_in               = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_in               = tf.Variable(tf.truncated_normal([self.number_of_hidden]))
        #
        #FC out
        h_out_shape             = (Lmnist.NUMBER_OF_CALSSES, self.number_of_hidden)
        self.h_out              = tf.Variable(tf.truncated_normal(h_out_shape))
        self.b_out              = tf.Variable(tf.truncated_normal([Lmnist.NUMBER_OF_CALSSES]))

    def build_model(self):
        self._locd.build_model()
        i2h = tf.nn.relu(tf.matmul(self.h_in, self._input) + tf.expand_dims(self.b_in, 1))
        self._logits = tf.reshape(tf.matmul(self.h_out, i2h) + tf.expand_dims(self.b_out, 1), [Lmnist.NUMBER_OF_CALSSES])
        self._pred = tf.nn.softmax(self._logits)


    @property
    def classify_loss(self):
        if self._classify_loss is None:
            self._classify_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits,targets=self._labels))
        return self._classify_loss

    @property
    def sparse_loss(self):
        return self._locd.loss

    @property
    def predict(self):
        if self._pred is None:
            self._pred = tf.nn.softmax(self._logits)
        return self._pred

    @property
    def labels(self):
        return self._labels

    @property
    def sparse_target(self):
        return self._locd.target

    @property
    def input(self):
        return self._locd.input

    @property
    def Z(self):
        return self._locd.output
    
            
            

