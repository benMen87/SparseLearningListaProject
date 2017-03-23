from lcod import LCoD
from lista import LISTA
import tensorflow as tf

class SCmnis(object):
    """Testing MNIST classification by passing full 28X28 through LCOD and then 
       into simple FC layer and then soft max all should learn to gather.
    """
     def __init__(self, input, input_size, batch_size, number_of_hidden=200)(self, We_shape, unroll_count, sc_block, number_of_hidden=200):
        
        #
        # pass full size
        self._input = self.lcod.output
        self._labels = tf.placeholder(tf.float32, shape=(1, num_labels))
        self.number_of_hidden = number_of_hidden
        self._pred = None
        self._logits = None
        #
        # model vars
        h_in_shape    = (self.number_of_hidden, self.lcod.output.get_shape()[0])
        self.h_in     = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_in     = tf.Variable(tf.truncated_normal(self.number_of_hidden))
        h_out_shape   = (NUMBER_OF_CALSSES, self.number_of_hidden)
        self.h_out    = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_out    = tf.Variable(tf.truncated_normal(NUMBER_OF_CALSSES))

        def build_model(self):
            self.lcod.build_model()
            i2h = tf.nn.relu(tf.matmul(self.h_in, self._input) + self.b_in)
            self._logits = tf.matmul(self.h_out, i2h) + self.b_out

        @property
        def loss(self):
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=self._labels)) \
                         + self.lcod.loss 

        @property
        def predict(self):
            self._pred = tf.nn.softmax(self._logits)


class FCmnist(object):
    """
    Very simple FC network for mnist classification
    """
    NUMBER_OF_CALSSES = 10

    def __init__(self, input, input_size, batch_size=None, number_of_hidden=200):
        """
        args:
            input::Can be tensorflow placholder or tensor output cascaded.
            input_size::Length of input tensor.
            batch_size::Mini batch size, Can be None.
            number_of_hidden::Amount of hidden units.
        """
        #
        # pass full size
        self._input = input
        self._labels = tf.placeholder(tf.float32, shape=(num_labels, 1))
        self.number_of_hidden = number_of_hidden
        self._pred = None
        self._logits = None
        #
        # model vars
        # L1
        h_in_shape = (input_size, self.number_of_hidden)
        self.h_in = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_in = tf.Variable(tf.truncated_normal(self.number_of_hidden))
        #
        # L2
        h_out_shape = (self.number_of_hidden, NUMBER_OF_CALSSES)
        self.h_out = tf.Variable(tf.truncated_normal(h_in_shape))
        self.b_out = tf.Variable(tf.truncated_normal(NUMBER_OF_CALSSES))

        def build_model(self):
            self.lcod.build_model()
            i2h = tf.nn.relu(tf.matmul(self._input, self.h_in) + self.b_in)
            self._logits = tf.matmul(i2h, self.h_out) + self.b_out

        @property
        def classify_loss(self):
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=self._labels))

        @property
        def predict(self):
            self._pred = tf.nn.softmax(self._logits)


