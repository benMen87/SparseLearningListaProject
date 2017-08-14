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
from Utils import load_fruit_images 
from Utils.psnr import psnr as psnr

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
parser.add_argument('--shirnkge_type', default='soft thresh',
                        choices=['soft thresh', 'smooth soft thresh'])
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model', dest='load_model', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--name', default='lista_ed', type=str, help='used for\
        creating load/store log dir names')
parser.add_argument('--load_name', default='', type=str, help='used to\
        load from a model with "name" diffrent from this model name')
parser.add_argument('--dataset', default='stl10', choices=['mnist', 'stl10', 'cifar10'])
parser.add_argument('--sparse_factor', '-sf',  default=0.5, type=float)
parser.add_argument('--middle_loss_factor', '-mf', default=0.1, type=float)
parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
parser.add_argument('--dont_train_dict', '-dt', action='store_true',  help='how many epochs to wait train dict -1 means dont train')
parser.add_argument('--test_fruits', action='store_true', help='test on fruit images for bechmark')
parser.add_argument('--add_noise',  action='store_true', help='add noise to input')
parser.add_argument('--inpaint',  action='store_true', help='add noise to input')
parser.add_argument('--inpaint_keep_prob', '-p', type=float, default=0.5,
        help='probilty to sample pixel')
args = parser.parse_args()

###########################################################
#                  Pre-Proccess Data Sets 
###########################################################
def rgb2gray(X):
    r, g, b = X[..., 0], X[...,1], X[...,2]
    return (0.2125 * r) + (0.7154 * g) + (0.0721 * b)

def add_noise(X, std):
    noise = np.random.normal(0, std, X.shape)
    return X + noise


def preprocess_data(X, addnoise=False, noise_sigma=0, inpaint=False, keep_prob=0.5):
    if addnoise:
       X = add_noise(X, noise_sigma)
    if inpaint:
        X *= np.random.choice([0, 1], size=X.shape, p=[1 - keep_prob, keep_prob])
    X = X.astype('float32')
    X -= np.mean(X, axis=(1, 2), keepdims=True)
    X /= np.std(X, axis=(1, 2), keepdims=True)
    X = X[..., np.newaxis]
    return X

###########################################################
#                   Load Data Sets
###########################################################



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

Y_train = np.empty(shape=X_train.shape)
Y_test =  np.empty(shape=X_test.shape)
Y_train[:] = X_train[:]
Y_test[:] = X_test[:]

noise_sigma = 20
# if args.add_noise == 1:
#     X_train = add_noise(X_train, noise_sigma)
#     X_test = add_noise(X_test, noise_sigma)

X_train = preprocess_data(X_train, args.add_noise, noise_sigma,
        args.inpaint, args.inpaint_keep_prob)
Y_train = preprocess_data(Y_train)

X_test = preprocess_data(X_test, args.add_noise, noise_sigma, args.inpaint, args.inpaint_keep_prob)
Y_test = preprocess_data(Y_test)


# split train set
X_valid = X_train[:1000]
X_train = X_train[1000:]
Y_valid = Y_train[:1000]
Y_train = Y_train[1000:]

######################################################
#   Plot and save test image
######################################################
def savetstfig(sess, encoder, decoder, inputim, targetim, fname):
    
    Xhat = decoder.recon_image()
    Z, im_hat = sess.run([encoder.output2d, Xhat], {encoder.input: inputim})
    
    example_ims = DIR_PATH + 'logdir/data/' + fname
    f, axarr = plt.subplots(2, 2)
    np.savez(example_ims, X=inputim, Y=targetim, Z=Z, IM_hat=im_hat)
 
    
    print('saved example img data de/en at %s' % example_ims)
   
    axarr[0, 1].axis('off')    
    
    axarr[0,0].imshow(np.squeeze(targetim), cmap='gray')
    axarr[0,0].set_title('original image')
    axarr[0,0].axis('off')

    axarr[1,1].imshow(np.squeeze(inputim), cmap='gray')
    axarr[1,1].set_title('noisy image -  psnr: {0:.3} [db]'.format(psnr(targetim, inputim)))
    axarr[1,1].axis('off')
     
    axarr[1,0].imshow(np.squeeze(im_hat), cmap='gray')
    axarr[1,0].set_title('reconstructed image -  psnr: {0:.3} [db]'.format(psnr(targetim, im_hat)))
    axarr[1,0].axis('off')

    example_ims = DIR_PATH + 'logdir/plots/' + fname + '.png'
    f.savefig(example_ims)
    print('saved example img plot de/en at %s' % example_ims)


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

# Define decoder class
class Decoder():
    def __init__(self, init_val, output_shape):
        self.decoder = tf.Variable(init_val, name='decoder', trainable=decoder_trainable)
        self.decoder = tf.nn.l2_normalize(self.decoder, dim=[0, 1], name='normilized_dict') # keep decoder atoms with l2 norm of 1
        self.target = tf.placeholder(tf.float32, shape=output_shape)
        self.output = []

    def reconstruct(self, encoder_out):
        return tf.nn.conv2d(encoder_out, self.decoder, strides=[1, 1, 1, 1], padding='SAME')

    def build_model(self, encoded_in):
         """Can pass multiple inputs at diffrent time state of encoder"""
         self.output = [self.reconstruct(enc_in) for enc_in in  encoded_in]

    def decoder_outall(self):
        return self.output

    def recon_image(self):
        return self.output[-1]

    def recon_loss_layer_i(self, input_index, name='l2'):
        if input_index > len(self.output):
            raise IndexError('layer index out of bounds')
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.target - self.output[input_index]), [1, 2]), name=name)



#
# build encoder-decoder
in_encoder_shape = input_shape[1] * input_shape[2]
out_encoder_shape = args.kernel_count * in_encoder_shape
We_shape = (out_encoder_shape, in_encoder_shape)

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
    encoder = sparse_encoder.LISTAConvDict2d(We_shape=We_shape,
                                             unroll_count=args.unroll_count,
                                             L=1, batch_size=args.batch_size,
                                             kernel_size=args.kernel_size,
                                             shrinkge_type=args.shirnkge_type,
                                             kernel_count=args.kernel_count,
                                             init_params_dict=init_en_dict)
encoder.build_model()
decoder_trainable = not args.dont_train_dict
with tf.variable_scope('decoder'):
    init_de = init_de if init_de is not None else encoder._Wd.initialized_value()
    output_shape =[None] + list(input_shape[1:]) + [1]
    decoder = Decoder(init_de, output_shape=output_shape)
    deco_input = [encoder.output2d_i(mdl_i), encoder.output2d]
    decoder.build_model(deco_input)

# l2-loss with index -1 is reconstruction of final encoding 
# l2-loss with index 0 is of indermidate encode
l_rec  = decoder.recon_loss_layer_i(-1, 'recon_loss') + \
         args.middle_loss_factor *  decoder.recon_loss_layer_i(0, 'recon_loss_stage_%d'%mdl_i)
loss = l_rec
#
# LOSS
if args.sparse_factor:
    l_sparse = tf.reduce_mean(tf.reduce_sum(tf.abs(encoder.output), 1), name='l1')
    loss += args.sparse_factor * l_sparse

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
    variable_summaries(decoder.decoder)
with tf.name_scope('sparse_code'):
    variable_summaries(encoder.output)

tf.summary.scalar('encoded_sparsity',
        tf.reduce_mean(tf.count_nonzero(encoder.output, 1)))
if args.sparse_factor:
    tf.summary.scalar('l1_loss', l_sparse)
tf.summary.scalar('l2recon_loss', l_rec)
tf.summary.scalar('total_loss', loss)
tf.summary.image('input', encoder.input)
tf.summary.image('output', decoder.recon_image())
tf.summary.image('target', decoder.target)

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
if not args.dont_train_dict:
    optimizer_de = tf.train.AdamOptimizer(1.5 * args.learning_rate).minimize(loss, var_list=decoder_vars, global_step=global_step_de)


def nextbatch(X, Y, batch_size=500, run_once=False):
    offset = 0
    data_len = X.shape[0]
    batch_Y = None # label
    batch_X = None # input image maybe with noise

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
test_batch = nextbatch(X=X_test, Y=Y_test, batch_size=500, run_once=True)

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
with tf.Session() as sess:

    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    merged = tf.summary.merge_all()
    train_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name)
    train_summ_writer.add_graph(sess.graph)
    if args.load_model:
        if args.load_name != '':
            LOAD_DIR = DIR_PATH + '/logdir/models/' + args.load_name + '/'
        else:
            LOAD_DIR = MODEL_DIR
        if os.listdir(LOAD_DIR):
            print('loading model')
            saver.restore(sess, tf.train.latest_checkpoint(LOAD_DIR))
        else:
            print('no cpk to load running with random init')

    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("all_zero", all_zero)

    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0
        print('epoch number:%d', epoch)
        train_batch = nextbatch(X=X_train, Y=Y_train, batch_size=args.batch_size,
                                run_once=True)
        b_num = 0
        for X_batch, Y_batch  in train_batch:
            b_num += 1
            np.savez('one_batch_save', X=X_batch, C=Y_batch)
            _, iter_loss = sess.run([optimizer_en, loss], {encoder.input: X_batch, decoder.target: Y_batch})
            if not  args.dont_train_dict:
                _, iter_loss = sess.run([optimizer_de, loss], {encoder.input: X_batch, decoder.target: Y_batch})
            train_loss.append(iter_loss)
            epoch_loss += iter_loss


            if b_num % 50 == 0:
                summary = sess.run(merged, {encoder.input: X_batch, decoder.target: Y_batch})
                train_summ_writer.add_summary(summary, global_step_en.eval())

            if b_num % 100 == 0:
                valid_loss = 0
                v_itr = 0
                sp_in_itr = 0
                sp_out_itr = 0
                l1 = 0
                l2 = 0
                vaild_batch = nextbatch(X=X_valid[:100], Y=Y_valid[:100], batch_size=50, run_once=True)

                for Xv_batch, Yv_batch in vaild_batch:
                    v_itr += 1
                    iter_loss, iter_l2, enc_out = sess.run([loss, l_rec, encoder.output],
                                                           {encoder.input: Xv_batch, decoder.target: Yv_batch})

                    sp_out_itr += np.count_nonzero(enc_out)/enc_out.shape[0]
                    sp_in_itr += np.count_nonzero(X_batch)/Xv_batch.shape[0]
                    valid_loss += iter_loss
                    l2 += iter_l2
                valid_loss /= v_itr
                l2 /= v_itr
                validation_loss.append(valid_loss)
                valid_sparsity_out.append(sp_out_itr/v_itr)
                valid_sparsity_in.append(sp_in_itr/v_itr)
                if args.save_model:
                    if valid_loss <= np.min(validation_loss):
                        f_name = MODEL_DIR + 'csc_u{}_'.format(args.unroll_count)
                        saver.save(sess, f_name, global_step=global_step_en)
                        print('saving model at: %s'%f_name) 
                print('valid loss: %f l2 loss: %f encoded sparsity: %f' %
                      (valid_loss, l2, valid_sparsity_out[-1]))

        print('epoch %d: loss val:%f' % (epoch, args.batch_size  * epoch_loss / X_train.shape[0]))
    
    test_loss = 0
    l2_loss = 0
    test_iter = 0
    for X_batch, Y_batch in test_batch:
        test_iter += 1
        test_loss += sess.run(loss, {encoder.input: X_batch, decoder.target: Y_batch})
        l2_loss += sess.run(l_rec, {encoder.input: X_batch, decoder.target: Y_batch})
    print('='*40)
    print('test loss: %f l2 loss: %f' % ((test_loss/test_iter), (l2_loss/test_iter)))
    
    # for debug save We/Wd
    We_Wd = DIR_PATH + 'logdir/data/We_Wd'
    We, Wd = sess.run([encoder._We, encoder._Wd])
    np.savez(We_Wd, Wd=Wd, We=We)

    if args.test_fruits:
        X_fruit = rgb2gray(load_fruit_images.load())
        fruit_std = np.mean(np.std(X_fruit, axis=(1, 2), keepdims=True))
        Y_fruit = np.empty(shape=X_fruit.shape)
        Y_fruit[:] = X_fruit[:]
        X_fruit = preprocess_data(X_fruit, args.add_noise, noise_sigma, args.inpaint, args.inpaint_keep_prob)
        Y_fruit = preprocess_data(Y_fruit)

        for i, (xfruit, yfruit)  in enumerate(zip(X_fruit, Y_fruit)):
            savetstfig(sess, encoder, decoder, xfruit[np.newaxis, :], yfruit[np.newaxis, :], 'fruit'+str(i))

    # plot example image
    for ex_i in range(20):
        i = np.random.randint(X_test.shape[0], size=1)
        x = X_test[i, :]
        y = Y_test[i, :]
        savetstfig(sess, encoder, decoder, x, y, 'example'+str(ex_i))               
    
    print('='*40)
    decoder_filters = decoder.decoder.eval()
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
plot_num = np.sqrt(args.kernel_count)
fid = 0
decoder_filters = np.squeeze(decoder_filters)
for f in decoder_filters.T:
    plt.subplot(plot_num, plot_num, fid+1)
    plt.imshow(f.T, interpolation='bilinear', cmap='gray')
    fid += 1
plt.savefig(DIR_PATH + 'logdir/plots/filters.png')
print('seved plot of dict filter atoms in {}'.format(DIR_PATH + 'logdir/plots/filters.png'))


