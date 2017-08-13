import tensorflow  as tf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../approx_sparse_coding'))
sys.path.append(os.path.abspath(DIR_PATH + '../'))

import lista_convdict2d as sparse_encoder


class ConvSparseDecoder(object):
    """
    Decode a lista based CSC image.
    """
    def __init__(self, init_filters):
        self.filters = tf.nn.l2_normalize(tf.Variable(init_filters, name='decoder'), dim=[0, 1], name='normilized_dict')
        self.out = None

    def build_model(self, input):
        self.out = tf.nn.conv2d(input, self.filters, strides=[1, 1, 1, 1], padding='SAME')
        return self.out



class ConvSparseAE(object):
    """
    This class uses a variation of LISTA with convolutions as the encoder.
    And a set of filter arrays for the decoder.
    """
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.inputshape = 0

    def create_model(self, in_shape, out_shape, unroll_count, batch_size,
                      kernel_size, shrinkge_func, kernel_count, load_pre_trained=False):
        """
        over here we create the convolutional sparse encoder (CSC)
        and convolutional sparse decoder instances 
        """
        self.inputshape = in_shape

        init_en_dict = {}
        if load_pretrained_dict:
            print('loading dict')
            trained_dict = np.load(DIR_PATH + 'pretrained_dict/learned_dict.npy')
            init_de = trained_dict.astype('float32')
            init_en_dict['Wd'] = init_de
            init_en_dict['We'] = 0.01 * np.transpose(np.flip(np.flip(init_de, axis=0), axis=1), [0, 1, 3, 2])



        with tf.variable_scope('encoder'):
            self.encoder = \
            sparse_encoder.LISTAConvDict2d(We_shape=We_(out_shape, in_shape),
                                           unroll_count=unroll_count,
                                           L=1, batch_size=batch_size,
                                           kernel_size=kernel_size,
                                           shrinkge_type=shirnkge_type,
                                           kernel_count=kernel_count,
                                           init_params_dict=init_en_dict)

        with tf.variable_scope('decoder'):
            init_de = self.encoder._Wd.initialized_value()
            self.decoder = ConvSparseDecoder(init_de)

    def build_model(self):
        self.encoder.build_model()
        self.decoder.build_model(self.encoder.output2d)

    @property
    def input2D(self):
        return self.encoder.input2D

    @proberty
    def input(self):
        return self.encoder.input

    @property
    def output2d():
        return decoder.out
