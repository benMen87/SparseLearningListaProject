from lcod import LCoD
from lista import LISTA
import tensorflow as tf


class Lmnist(object):
    """Testing MNIST classification by passing full 28X28 through LCOD and then 
       into simple FC layer and then soft max all should learn to gather.
    """
    NUMBER_OF_CALSSES = 10

    def __init__(self, We_shape, unroll_count,
                 sc_type='lista', batch_size=None,
                 number_of_hidden=200, We=None):

        if sc_type == 'lcod':
            self._sc_block = LCoD(We_shape, unroll_count, We
                                  shrinkge_type='soft thresh',
                                  batch_size=batch_size)
        elif sc_type == 'list':
            self._sc_block = LISTA(W_shape, unroll_count, We,
                                   shrinkge_type='soft thresh',
                                   batch_size=batch_size)
        #
        # model vars
        self._input = None
        self._labels = tf.placeholder(tf.float32, shape=(batch_size,
                                                         self.NUMBER_OF_CALSSES))
        self.number_of_hidden = number_of_hidden
        self._pred = None
        self._logits = None
        self._classify_loss = None
        #
        # FC in
        h_in_shape = (self._sc_block.output_size, self.number_of_hidden)
        self.h_in = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_in = tf.Variable(tf.truncated_normal([self.number_of_hidden]))
        #
        # FC out
        h_out_shape = (self.number_of_hidden, NUMBER_OF_CALSSES)
        self.h_out = tf.Variable(tf.truncated_normal(h_out_shape))
        self.b_out = tf.Variable(tf.truncated_normal([self.NUMBER_OF_CALSSES]))

    def build_model(self):
        self._sc_block.build_model()
        self._input = self._sc_block.output
        i2h = tf.nn.relu(tf.matmul(self._input, self.h_in) + self.b_in)
        self._logits = tf.matmul(i2h, self.h_out) + self.b_out
        self._pred = tf.nn.softmax(self._logits, dim=0)

    @property
    def classify_loss(self):
        if self._classify_loss is None:
            self._classify_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=self._logits, targets=self._labels)
            self._classify_loss = tf.reduce_mean(self.classify_loss)

        return self._classify_loss

    @property
    def sparse_loss(self):
        return self._sc_block.loss

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
        return self._sc_block.target

    @property
    def input(self):
        return self._sc_block.input

    @property
    def Z(self):
        return self._sc_block.output


