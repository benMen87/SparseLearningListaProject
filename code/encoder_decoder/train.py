import os
import sys
sys.path.append('/home/benmen/Msc3yr/Sparse/SparseLearningListaProject/code/approx_sparse_coding')

import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lista_convdict2d as sparse_encoder

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'

parser = argparse.ArgumentParser(description='Sparse encoder decoder model')


parser.add_argument('-b', '--batch_size', default=50,
                    type=int, help='size of train batches')
parser.add_argument('-n', '--num_steps', default=1, type=int,
                    help='number of training steps')
parser.add_argument('-ks', '--kernel_size', default=3, type=int,
                    help='kernel size to be used in lista_conv')
parser.add_argument('-kc', '--kernel_count', default=64, type=int,
                    help='amount of kernel to use in lista_conv')
parser.add_argument('-u', '--unroll_count', default=7,
                    type=int,
                    help='Amount of Reccurent time steps for decoder')
parser.add_argument('-o', '--output_dir_path',
                    default='',
                    type=str,
                    help='output directory to save model if non is given\
                          model wont be saved')
args = parser.parse_args()


###########################################################
#                   Load Data Sets
###########################################################

#
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("training size: {}, test size: {}".format(y_train.shape[0],
      y_test.shape[0]))

# flatten for encoder
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# normilize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# split train set
(X_valid, Y_valid) = (X_train[:5000], Y_train[:5000])
(X_train, Y_train) = (X_train[5000:], Y_train[5000:])

########################################################
#       Build Models - Sparse Encoder Decoder
########################################################

#
# build encoder-decoder
in_encoder_shape = 28 * 28
out_encoder_shape = args.kernel_count * in_encoder_shape
We_shape = (out_encoder_shape, in_encoder_shape)

with tf.variable_scope('encoder'):
    encoder = sparse_encoder.LISTAConvDict2d(We_shape=We_shape,
                                             unroll_count=args.unroll_count,
                                             L=8, batch_size=args.batch_size,
                                             kernel_size=args.kernel_size,
                                             kernel_count=args.kernel_count)
encoder.build_model()
with tf.variable_scope('decoder'):
    init_de = tf.random_normal([args.kernel_size, args.kernel_size,
                                args.kernel_count, 1])
    D = tf.Variable(init_de)
    Xhat = tf.nn.conv2d(encoder.output2D, D, strides=[1, 1, 1, 1], padding='SAME')

loss = tf.reduce_mean(tf.square(encoder.input2D - Xhat)) + \
       0.5 / 8 * tf.reduce_sum(tf.abs(encoder.output))

#######################################################
#   Training Vars - optimizers and batch generators
#######################################################
encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder/")
decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder/")
optimizer_en = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=encoder_vars)
optimizer_de = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=decoder_vars)



def nextbatch(X, Y, batch_size, run_once=False):
    offset = 0
    data_len = X.shape[0]
    while True:
        if offset + batch_size <= data_len:
            batch_X = X[offset: batch_size + offset]
            batch_Y = Y[offset: batch_size + offset]
            offset = offset + batch_size
        else:
            if run_once:
                raise StopIteration()

            batch_X = np.concatenate((X[offset: data_len], X[:batch_size - (data_len - offset)]), axis=0)
            batch_Y = np.concatenate((Y[offset: data_len], Y[:batch_size - (data_len - offset)]), axis=0)
            offset = batch_size - (data_len - offset)
        yield X, Y


train_batch = nextbatch(X_train, Y_train, args.batch_size)
vaild_batch = nextbatch(X_valid, Y_valid, 500)
test_batch = nextbatch(X_test, Y_test, 500, run_once=True)

###################################################################
#                Training   +   Results 
###################################################################

train_loss = []
validation_loss = []
epoch_loss = 0
test_loss = 0
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    for iter in range(args.num_steps):

        X_batch, _ = train_batch.next()

        if iter % 10 == 0:
            _, iter_loss = sess.run([optimizer_de, loss], {encoder.input: X_batch})
        else:
            _, loss = sess.run([optimizer_en, loss], {encoder.input: X_batch})
        train_loss.append(iter_loss)

        epoch_loss += iter_loss
        if iter % X_test.shape[0] == 0:
            print('epoch %d: loss val:%f'%(iter//X_test.shape[0], epoch_loss / X_test.shape[0]))
            epoch_loss = 0

        if iter % 500:
            print('train iter %d loss %f'%(iter, iter_loss))

        if iter % 1000 == 0:
            print('cross validation')
            X_batch, _ = vaild_batch.next()
            iter_loss =  sess.run(loss, {encoder.input: X_batch})
            print('valid loss on randome batch of 500 imgs: %f'%iter_loss)
            validation_loss.append(iter_loss)

        for X_batch, _ in test_batch:
            print('tst')
            test_loss += sess.run(loss, {encoder.input: X_batch})
        print('test loss: %f'%(test_loss/y_test.shape[0]))

        decoder_filters = D.eval()
        decoder_filters_path = DIR_PATH + 'logdir/data/decoder_filters'
        np.save(decoder_filters_path, decoder_filters)
        print('saved decoder filters at path: %s'%decoder_filters_path)


plt.figure()
plt.plot(args.num_steps, train_loss, 'r', label='test loss')
plt.plot(args.num_steps//1000, validation_loss, 'g', label='validation loss' )
plt.ylabel('loss')
plt.xlabel('#iter')
plt.legend(loc='upper right')
plt.title('loss = ||decode(encode(X)) - X||_2^2 + lamda||encode(X)||_1')
fig_path = DIR_PATH + 'logdir/plots/validtrainloss'
plt.savefig(fig_path)
print('plot saved in path: %s'%fig_path)
