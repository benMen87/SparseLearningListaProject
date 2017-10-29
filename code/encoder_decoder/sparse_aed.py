import os
import sys
import numpy as np
import tensorflow as tf


DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../approx_sparse_coding'))
sys.path.append(os.path.abspath(DIR_PATH + '../'))

import lista_convdict2d as sparse_encoder 


# Define decoder class
class Decoder():
    def __init__(self, init_val, output_shape):
        self.decoder = tf.Variable(init_val, name='decoder') 
        self.decoder = tf.nn.l2_normalize(self.decoder, dim=[0, 1], name='normilized_dict') # keep decoder atoms with l2 norm of 1
        self.target = tf.placeholder(tf.float32, shape=output_shape)
        self.output = []

    def reconstruct(self, encoder_out):
        return tf.nn.conv2d(encoder_out, self.decoder, strides=[1, 1, 1, 1],
                padding='SAME')

    def build_model(self, encoded_in):
        """Can pass multiple inputs at diffrent time state of encoder"""
        self.output = [self.reconstruct(enc_in) for enc_in in encoded_in]

    def decoder_outall(self):
        return self.output

    def recon_image(self):
        return self.output[-1]

    def recon_loss_layer_i(self, input_index, mask=1,  dist_name='l2', name='recon_loss'):
        if input_index > len(self.output):
            raise IndexError('layer index out of bounds')
        if dist_name == 'l2':
            dist_fn = tf.square
        elif dist_name == 'l1':
            dist_fn = tf.abs

        boundry_mask = np.zeros(self.target.shape[1:4])
        boundry = self.decoder.get_shape().as_list()[0] // 2
        boundry_mask[boundry:-boundry, boundry:-boundry, :] = 1.0
        boundry_mask= tf.constant(boundry_mask.astype('float32'))

        recon = self.output[input_index]
        dist = dist_fn(self.target - recon)
        return tf.reduce_mean(tf.reduce_sum(boundry_mask * dist, [1, 2]), name=name)

def build_model(args, input_shape):

    #
    # build encoder-decoder
    
    init_de = None
    init_en_dict = {}
    if args.load_pretrained_dict:
        print('loading dict')
        trained_dict = np.load(DIR_PATH + 'pretrained_dict/learned_dict.npy')
        init_de = trained_dict.astype('float32')
        init_en_dict['Wd'] = init_de
        init_en_dict['We'] = 0.01 * np.transpose(np.flip(np.flip(init_de, axis=0), axis=1), [0, 1, 3, 2])

    mdl_i = args.unroll_count // 2

    with tf.variable_scope('encoder'):
        encoder = sparse_encoder.LISTAConvDict2d(inputshape=input_shape,
                                                 unroll_count=args.unroll_count,
                                                 L=1, batch_size=args.batch_size,
                                                 kernel_size=args.kernel_size,
                                                 shrinkge_type=args.shrinkge_type,
                                                 kernel_count=args.kernel_count,
                                                 init_params_dict=init_en_dict,channel_size=input_shape[-1])
        if args.inpaint:
            encd_mask = tf.placeholder(tf.float32, shape=encoder.input2D.shape)
        else:
            encd_mask = 1 # np.zeros(input_shape)
            #edges = args.kernel_size // 2
            #encd_mask[edges:-edges, edges:-edges, :] = 1.0
            #encd_mask = tf.constant(encd_mask.astype('float32'))

    encoder.build_model(encd_mask)
    decoder_trainable = not args.dont_train_dict
    with tf.variable_scope('decoder'):
        init_de = encoder.Wd.initialized_value()
        output_shape =[None] + list(input_shape)
        decoder = Decoder(init_de, output_shape=output_shape)
        deco_input = [encoder.output]
        decoder.build_model(deco_input)

    return encoder, decoder, encd_mask
