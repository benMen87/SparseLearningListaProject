import os
import sys
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.python import debug as tf_debug
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../approx_sparse_coding'))
sys.path.append(os.path.abspath(DIR_PATH + '../'))
import lista_convdict2d as sparse_encoder
from Utils import stl10_input


parser = argparse.ArgumentParser(description='Sparse encoder decoder model')

parser.add_argument('-b', '--batch_size', default=15,
                            type=int, help='size of train batches')
parser.add_argument('-n', '--num_epochs', default=5, type=int,
                            help='number of epochs steps')
parser.add_argument('-ks', '--kernel_size', default=5, type=int,
                            help='kernel size to be used in lista_conv')
parser.add_argument('-kc', '--kernel_count', default=2, type=int,
                            help='amount of kernel to use in lista_conv')
parser.add_argument('-u', '--unroll_count', default=5,
                    type=int, help='Amount of Reccurent timesteps for decoder')
parser.add_argument('--shirnkge_type', default='smooth soft thresh',
                        choices=['soft thresh', 'smooth soft thresh'])
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model', dest='load_model', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--name', default='lista_ed', type=str)
parser.add_argument('--dataset', default='stl10', choices=['mnist', 'stl10', 'cifar10'])
parser.add_argument('--sparse_factor', '-sf',  default=0.5, type=float)

args = parser.parse_args()

###########################################################
#                   Load Data Sets
###########################################################


def rgb2gray(X):
    r, g, b = X[..., 0], X[...,1], X[...,2]
    return (0.2125 * r) + (0.7154 * g) + (0.0721 * b)

#
# load data
if args.dataset == 'mnist':
    (X_train, _), (X_test, _) = mnist.load_data()
elif args.dataset == 'stl10':  # Data set with unlabeld too large cant loadfully to memory
    (X_train, _), (X_test, _), X_unlabel = stl10_input.load_data(grayscale=True)
    if X_unlabel.shape[0] > 0:
        X_train = np.concatenate((X_train, X_unlabel), axis=0)
    np.random.shuffle(X_train)
    np.random.shuffle(X_test)
elif args.dataset == 'cifar10':
    (X_train, _), (X_test, _) = cifar10.load_data()
    X_train, X_test = rgb2gray(X_train), rgb2gray(X_test)


print("training size: {}, test size: {}".format(X_train.shape[0],
      X_test.shape[0]))

input_shape = X_train.shape
print('shape {}'.format(input_shape))

# flatten for encoder
X_train = X_train.reshape(X_train.shape[0], input_shape[1] * input_shape[2])
X_test = X_test.reshape(X_test.shape[0], input_shape[1] * input_shape[2])

# normilize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train  -= np.mean(X_train, axis=1, keepdims=True)
X_train /= np.std(X_train, axis=1, keepdims=True)
X_test  -= np.mean(X_test, axis=1, keepdims=True)
X_test /= np.std(X_test, axis=1, keepdims=True)

# one hot
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)

# split train set
X_valid = X_train[:1000]
X_train = X_train[1000:]

#######################################################
#   Log-network for tensorboard
#######################################################
tensorboard_path = DIR_PATH + 'logdir/tb/'

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

########################################################
#       Build Models - Sparse Encoder Decoder
########################################################

#
# build encoder-decoder
in_encoder_shape = input_shape[1] * input_shape[2]
out_encoder_shape = args.kernel_count * in_encoder_shape
We_shape = (out_encoder_shape, in_encoder_shape)

# output = DIR_PATH + 'logdir/data/'
# if args.load_model:
#     print('loading model from %s' % output)
#     init_dict = np.load(output + 'encoder')
#     init_de = np.load(output + 'decoder')
# else:
init_dict = {}
init_de = None

with tf.variable_scope('encoder'):
    encoder = sparse_encoder.LISTAConvDict2d(We_shape=We_shape,
                                             unroll_count=args.unroll_count,
                                             L=1, batch_size=args.batch_size,
                                             kernel_size=args.kernel_size,
                                             shrinkge_type=args.shirnkge_type,
                                             kernel_count=args.kernel_count,
                                             init_params_dict=init_dict)
encoder.build_model()
with tf.variable_scope('decoder'):
    init_de = init_de if init_de is not None else encoder._Wd.initialized_value()
    D = tf.nn.l2_normalize(tf.Variable(init_de, name='decoder'), dim=[0, 1], name='normilized_dict')
    Xhat = tf.nn.conv2d(encoder.output2d, D, strides=[1, 1, 1, 1], padding='SAME')
#
# LOSS
l_rec = tf.reduce_mean(tf.reduce_sum(tf.square(encoder.input2D - Xhat), [1, 2]), name='l2')
l_sparse = tf.reduce_mean(tf.reduce_sum(tf.abs(encoder.output), 1), name='l1') 
loss =  l_rec + args.sparse_factor * l_sparse

#######################################################
# Add vars to summary
#######################################################
with tf.name_scope('encoder'):
    with tf.name_scope('We'):
        variable_summaries(encoder._We)
    with tf.name_scope('Wd'):
        variable_summaries(encoder._Wd)
    for i, t in enumerate(encoder._theta):
        with tf.name_scope('theta'+str(i)):
            variable_summaries(t)
    # for i, t in enumerate(encoder._b):
    #     with tf.name_scope('b'+str(i)):
    #         variable_summaries(t)
with tf.name_scope('decoder'):
    variable_summaries(D)
with tf.name_scope('sparse_code'):
    variable_summaries(encoder.output)

tf.summary.scalar('encoded_sparsity',
        tf.reduce_mean(tf.count_nonzero(encoder.output, 1)))
tf.summary.scalar('l1_loss', l_sparse)
tf.summary.scalar('l2recon_loss', l_rec)
tf.summary.scalar('total_loss', loss)
tf.summary.image('input', encoder.input2D, max_outputs=3)
tf.summary.image('output', Xhat, max_outputs=3)

#######################################################
# Saver
#######################################################
if args.save_model or args.load_model:
    MODEL_DIR = DIR_PATH + '/logdir/models/' + args.name + '/'
    saver = tf.train.Saver(max_to_keep=1)
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

#######################################################
#   Training Vars - optimizers and batch generators
#######################################################
global_step_en = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_en')
global_step_de = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_de')
encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder/")
decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder/")
optimizer_en = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, var_list=encoder_vars, global_step=global_step_en)
optimizer_de = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, var_list=decoder_vars, global_step=global_step_de)

def nextbatch(X, Y=None, batch_size=500, run_once=False):
    offset = 0
    data_len = X.shape[0]
    batch_Y = None
    batch_X = None
    while True:
        if offset + batch_size <= data_len:
            batch_X = X[offset: batch_size + offset]
            if Y is not None:
                batch_Y = Y[offset: batch_size + offset]
            offset = offset + batch_size
        else:
            if run_once:
                raise StopIteration()
            batch_X = np.concatenate((X[offset: data_len], X[:batch_size - (data_len - offset)]), axis=0)
            if Y is not None:
                batch_Y = np.concatenate((Y[offset: data_len], Y[:batch_size - (data_len - offset)]), axis=0)
            offset = batch_size - (data_len - offset)
        yield batch_X, batch_Y

print('batch size {}'.format(args.batch_size))
test_batch = nextbatch(X=X_test, Y=None, batch_size=500, run_once=True)

###################################################################
#                Training   +   Results
###################################################################
def all_zero(datum, tensor):
    return np.count_nonzero(tensor) == 0

train_loss = []
validation_loss = []
valid_sparsity_out = []
valid_sparsity_in = []
epoch_loss = 0
test_loss = 0
with tf.Session() as sess:

    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    merged = tf.summary.merge_all()
    train_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name)
    train_summ_writer.add_graph(sess.graph)
    if args.load_model:
        if os.listdir(MODEL_DIR):
            print('loading model')
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        else:
            print('no cpk to load running with random init')

    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("all_zero", all_zero)

    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0
        print('epoch number:%d', epoch)

        train_batch = nextbatch(X=X_train, Y=None, batch_size=args.batch_size, run_once=True)
        b_num = 0
        for X_batch, _  in train_batch:
            b_num += 1
            _, iter_loss  = sess.run([optimizer_en, loss], {encoder.input: X_batch})
            if epoch  > 1:
                _, iter_loss  = sess.run([optimizer_de, loss], {encoder.input: X_batch})
            train_loss.append(iter_loss)
            epoch_loss += iter_loss
            if b_num % 100 == 0:
                valid_loss = 0
                v_itr = 0
                sp_in_itr = 0
                sp_out_itr = 0
                l1 = 0

                vaild_batch = nextbatch(X=X_valid[:100], batch_size=50, run_once=True)
                for X_batch, _ in vaild_batch:
                    v_itr += 1
                    iter_loss, iter_l1, enc_out = sess.run([loss, l_sparse, encoder.output], {encoder.input: X_batch})
                    sp_out_itr += np.count_nonzero(enc_out)/enc_out.shape[0]
                    sp_in_itr += np.count_nonzero(X_batch)/X_batch.shape[0]
                    valid_loss += iter_loss
                    l1 += iter_l1
                valid_loss /= v_itr
                l1 /= v_itr
                validation_loss.append(valid_loss)
                valid_sparsity_out.append(sp_out_itr/v_itr)
                valid_sparsity_in.append(sp_in_itr/v_itr)
                if args.save_model:
                    if valid_loss <= np.min(validation_loss):
                        f_name = MODEL_DIR + 'csc_u{}_'.format(args.unroll_count)
                        saver.save(sess, f_name, global_step=global_step_en)
                        print('saving model at: %s'%f_name) 
                print('valid loss: %f l1 loss: %f encoded sparsity: %f' %
                      (valid_loss, l1, valid_sparsity_out[-1]))
            if b_num % 50 == 0:
                summary = sess.run(merged, {encoder.input: X_batch})
                train_summ_writer.add_summary(summary, epoch)

        print('epoch %d: loss val:%f' % (epoch, args.batch_size  * epoch_loss / X_train.shape[0]))

    test_iter = 0
    for X_batch, _ in test_batch:
        test_iter += 1
        test_loss += sess.run(loss, {encoder.input: X_batch})
    print('='*40)
    print('test loss: %f' % (test_loss/test_iter))

    # plot example image
    for ex_i in range(5):
        i = np.random.randint(X_test.shape[0], size=1)
        im = X_test[i, :]
        im_n = im +  np.random.normal(0, 0.1, input_shape[1] * input_shape[2])
        Z_n, im_hat_n = sess.run([encoder.output2d, Xhat], {encoder.input: im_n})
        Z, im_hat = sess.run([encoder.output2d, Xhat], {encoder.input: im})

        im.shape = (input_shape[1], input_shape[2])
        im_n.shape = (input_shape[1], input_shape[2])
        im_hat.shape = im.shape
        im_hat_n.shape = im_n.shape

        example_ims = DIR_PATH + 'logdir/data/example_im' + str(ex_i)
        np.savez(example_ims, IM=im, IM_n=im_n, Z=Z, Z_n=Z_n,  IM_hat=im_hat,
                IM_hat_n=im_hat_n)

        print('saved example img data de/en at %s' % example_ims)

        plt.figure()
        plt.subplot(221)
        plt.imshow(im, cmap='gray')
        plt.title('original image')
        plt.subplot(222)
        plt.imshow(im_hat, cmap='gray')
        plt.title('reconstructed image')
        plt.subplot(223)
        plt.imshow(im_n, cmap='gray')
        plt.title('noisy image')
        plt.subplot(224)
        plt.imshow(im_hat_n, cmap='gray')
        plt.title('reconstructed noisy image')

        example_ims = DIR_PATH + 'logdir/plots/example_im{}'.format(ex_i)
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
plt.plot(range(len(valid_sparsity_out)),  valid_sparsity_out, 'r', label='sparsity of encoded  image')
plt.plot(range(len(valid_sparsity_in)), valid_sparsity_in, 'g', label='sparsity of original image')
plt.legend(loc='upper right')
sc_plot = DIR_PATH + 'logdir/plots/sparsity.png'
plt.savefig(sc_plot)
np.savez(DIR_PATH + 'logdir/data/sparse', IN=valid_sparsity_in, EN=valid_sparsity_out)
print('plot of sparsity input vs encode in %s' % sc_plot)

plt.figure()
fid = 0
decoder_filters = np.squeeze(decoder_filters)
for f in decoder_filters.T:
    plt.subplot(8, 8, fid+1)
    plt.imshow(f.T, interpolation='bilinear', cmap='gray')
    fid += 1
plt.savefig(DIR_PATH + 'logdir/plots/filters.png')
print('seved plot of dict filter atoms in {}'.format(DIR_PATH + 'logdir/plots/filters.png'))


