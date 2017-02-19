from lcod import LCoD
import tensorflow as tf


class mnist(object):
    """Testing MNIST classification by passing full 28X28 through LCOD and then 
       into simple FC layer and then soft max all should learn to gather.
    """
    NUMBER_OF_CALSSES = 10
    def __init__(self, We_shape, unroll_count, number_of_hidden=200):

        self.lcod             = LCoD(We_shape, unroll_count)
        #
        # pass full size
        self._input           = self.lcod.output
        self._labels           = tf.placeholder(tf.float32, shape=(1, num_labels))
        self.number_of_hidden = number_of_hidden
        self._pred             = None
        self._logits            = None
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
            
            

