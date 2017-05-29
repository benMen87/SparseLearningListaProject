import os
import sys
import tensorflow as tf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../approx_sparse_coding'))

import argparse
import numpy as np
from tensorflow.python import debug as tf_debug
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

import lista_convdict2d as sparse_encoder

parser = argparse.ArgumentParser(description='Sparse encoder decoder model')


parser.add_argument('-b', '--batch_size', default=2,
                    type=int, help='size of train batches')
parser.add_argument('-n', '--num_epochs', default=30000, type=int,
                    help='number of epochs steps')
parser.add_argument('-ks', '--kernel_size', default=3, type=int,
                    help='kernel size to be used in lista_conv')
parser.add_argument('-kc', '--kernel_count', default=2, type=int,
                    help='amount of kernel to use in lista_conv')
parser.add_argument('-u', '--unroll_count', default=5,
                    type=int,
                    help='Amount of Reccurent time steps for decoder')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model', dest='load_model', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')


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
X_train  -= np.mean(X_train, axis=1, keepdims=True)
X_train /= np.std(X_train, axis=1, keepdims=True)
X_test  -= np.mean(X_test, axis=1, keepdims=True)
X_test /= np.std(X_test, axis=1, keepdims=True)

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

output = DIR_PATH + 'logdir/data/'
if args.load_model:
    print('loading model from %s' % output)
    init_dict = np.load(output + 'encoder')
    init_de = np.load(output + 'decoder')
else:
    init_dict = {}
    init_de = tf.truncated_normal([args.kernel_size, args.kernel_size,
                                args.kernel_count, 1], stddev=0.05)

with tf.variable_scope('encoder'):
    encoder = sparse_encoder.LISTAConvDict2d(We_shape=We_shape,
                                             unroll_count=args.unroll_count,
                                             L=8, batch_size=args.batch_size,
                                             kernel_size=args.kernel_size,
                                             kernel_count=args.kernel_count,
                                             init_params_dict=init_dict)
encoder.build_model()

with tf.variable_scope('decoder'):
    D = tf.Variable(init_de, name='decoder')
    Xhat = tf.nn.conv2d(encoder.output2D, D, strides=[1, 1, 1, 1], padding='SAME')
l_rec = tf.reduce_mean(tf.reduce_sum(tf.square(encoder.input2D - Xhat), [1, 2]))
l_sparse = tf.reduce_mean(tf.reduce_sum(tf.abs(encoder.output), 1))  
loss =  l_rec + 0.5 * l_sparse
       

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
        yield batch_X, batch_Y

print('batch size {}'.format(args.batch_size))
test_batch = nextbatch(X_test, Y_test, 500, run_once=True)

###################################################################
#                Training   +   Results
###################################################################


def all_zero(datum, tensor):
    return np.count_nonzero(tensor) == 0

train_loss = []
validation_loss = []
validation_spacity = []
epoch_loss = 0
test_loss = 0
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    print('Initialized')
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("all_zero", all_zero)

    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0
        train_batch = nextbatch(X_train, Y_train, args.batch_size, run_once=True)
        b_num = 0
        for X_batch, _  in train_batch:
            b_num += 1
            _, iter_loss = sess.run([optimizer_en, loss], {encoder.input: X_batch})
            if epoch  > 10:
                _, iter_loss  = sess.run([optimizer_de, loss], {encoder.input: X_batch})

            train_loss.append(iter_loss)

            epoch_loss += iter_loss
            if b_num % 500 == 0:
                ret_list = [encoder._Wd, encoder._We, encoder.output,encoder._theta,  D,  l_rec, l_sparse, encoder.output, Xhat]
                _Wd, _We, Z, _theta,  _D, ll_rec, ll_sparse, Z, XX = sess.run(ret_list, {encoder.input: X_batch})
                print('batch num %d in epoch %d loss %f' % (b_num, epoch, iter_loss))
                print('sanity check:')
                print('encoder/decoder abs max vals: Wd-{}, We-{} D-{}'.format(np.max(np.abs(_Wd)), np.max(np.abs(_We)), np.max(np.abs(_D))))
                print('max thata: {} encoder sparse: {}'.format(np.max(np.abs(_theta)), np.count_nonzero(Z)))

            if b_num % 200 == 0:
                print('cross validation')
                vaild_batch = nextbatch(X_valid, Y_valid, 500, run_once=True)
                valid_loss = 0
                valid_sparsity_out = []
                valid_sparsity_in = []
                v_itr = 0
                sp_in_itr = 0
                sp_out_itr = 0
                for X_batch, _ in vaild_batch:
                    v_itr += 1
                    iter_loss, enc_out = sess.run([loss, encoder.output], {encoder.input: X_batch})
                    sp_out_itr += np.count_nonzero(enc_out)/enc_out.shape[0]
                    sp_in_itr += np.count_nonzero(X_batch)/X_batch.shape[0]
                    valid_loss += iter_loss
                valid_loss /= v_itr
                validation_loss.append(valid_loss)
                valid_sparsity_out.append(sp_out_itr/v_itr)
                valid_sparsity_in.append(sp_in_itr/v_itr)
                print('valid loss: %f encoded sparsity: %f' % (valid_loss, valid_sparsity_out[-1]))
        print('epoch %d: loss val:%f' % (epoch, args.batch_size  * epoch_loss / X_train.shape[0]))

    test_iter = 0
    for X_batch, _ in test_batch:
        test_iter += 1
        test_loss += sess.run(loss, {encoder.input: X_batch})
    print('='*40)
    print('test loss: %f' % (test_loss/test_iter))

    if args.save_model:
        output = DIR_PATH + 'logdir/data/'
        print('saving model to %s' % output)
        We = encoder.We.eval()
        theta = sess.run(encoder._theta)
        Wd = encoder.Wd.eval()
        np.savez(output + 'encoder', We=We, theta=theta, Wd=Wd)
        np.savez(output + 'decoder', D=D.eval())

    # plot example image
    i = np.random.randint(X_test.shape[0], size=1)
    im = X_test[i, :]
    Z, im_hat = sess.run([encoder.output2D, Xhat], {encoder.input: im})
    example_ims = DIR_PATH + 'logdir/data/example_im'
    np.savez(example_ims, IM=np.reshape(im, (28, 28)), Z=Z, IM_hat=np.reshape(im_hat, (28, 28)))
    print('saved example img data de/en at %s' % example_ims)

    plt.figure()
    plt.subplot(211)
    plt.imshow(np.reshape(im, (28, 28)), cmap='gray')
    plt.title('original image')
    plt.subplot(212)
    plt.imshow(np.reshape(im_hat, (28, 28)), cmap='gray')
    plt.title('reconstructed image')
    example_ims = DIR_PATH + 'logdir/plots/example_im'
    plt.savefig(example_ims)
    print('saved example img plot de/en at %s' % example_ims)

    print('='*40)
    decoder_filters = D.eval()
    decoder_filters_path = DIR_PATH + 'logdir/data/decoder_filters'
    np.save(decoder_filters_path, decoder_filters)
    print('saved decoder filters at path: %s' % decoder_filters_path)

plt.figure()
plt.plot(range(len(train_loss) - 1), train_loss[1:])
plt.ylabel('loss')
plt.xlabel('#iter')
plt.title('train loss')
fig_path = DIR_PATH + 'logdir/plots/trainloss'
plt.savefig(fig_path)
print('plot saved in path: %s' % fig_path)

plt.figure()
plt.plot(range(len(validation_loss) - 1), validation_loss[1:])
plt.ylabel('loss')
plt.xlabel('#iter')
fig_path = DIR_PATH + 'logdir/plots/validloss'
plt.savefig(fig_path)
print('plot saved in path: %s' % fig_path)

plt.figure()
plt.plot(range(len(valid_sparsity_out)), valid_sparsity_out, 'r', label='sparsity of encoded  image')
plt.plot(range(len(valid_sparsity_in)), valid_sparsity_in, 'g', label='sparsity of original image')
plt.legend(loc='upper right')
sc_plot = DIR_PATH + 'logdir/plots/sparsity.png'
plt.savefig(sc_plot)
print('plot of sparsity input vs encode in %s' % sc_plot)

plt.figure()
fid = 0
decoder_filters = np.squeeze(decoder_filters)
for f in decoder_filters.T:
    plt.subplot(8, 8, fid+1)
    plt.imshow(f.T, cmap='gray')
    fid += 1
plt.savefig(DIR_PATH + 'logdir/plots/filters.png')
print('seved plot of dict filter atoms in {}'.format(DIR_PATH + 'logdir/plots/filters.png'))


