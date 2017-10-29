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
import scipy.io as scio
DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../approx_sparse_coding'))
sys.path.append(os.path.abspath(DIR_PATH + '../'))

import sparse_aed
from Utils import stl10_input
from Utils import load_images 
from Utils import load_berkeley
from Utils import ms_ssim
from Utils.psnr import psnr as psnr

parser = argparse.ArgumentParser(description='Sparse encoder decoder model')

parser.add_argument('-b', '--batch_size', default=2,
                            type=int, help='size of train batches')
parser.add_argument('-n', '--num_epochs', default=0, type=int,
                            help='number of epochs steps')
parser.add_argument('-ks', '--kernel_size', default=5, type=int,
                            help='kernel size to be used in lista_conv')
parser.add_argument('-kc', '--kernel_count', default=36, type=int,
                            help='amount of kernel to use in lista_conv')
parser.add_argument('--dilate', '-dl', action='store_true')
parser.add_argument('-u', '--unroll_count', default=10,
                    type=int, help='Amount of Reccurent timesteps for decoder')
parser.add_argument('--shrinkge_type', default='soft thresh',
                        choices=['soft thresh', 'smooth soft thresh'])
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model', dest='load_model', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--name', default='lista_ed', type=str, help='used for\
        creating load/store log dir names')
parser.add_argument('--load_name', default='', type=str, help='used to\
        load from a model with "name" diffrent from this model name')
parser.add_argument('--dataset', default='city_fruit') #'mnist','stl10', 'cifar10', 'pascal'
parser.add_argument('--whiten', action='store_true')
parser.add_argument('--sparse_factor', '-sf',  default=0.5, type=float)
parser.add_argument('--total_variation', '-tv',  default=0.0, type=float)
parser.add_argument('--recon_factor', '-rf',  default=1.0, type=float)
parser.add_argument('--ms_ssim', '-ms',  default=0.0, type=float)
parser.add_argument('--middle_loss_factor', '-mf', default=0.1, type=float)
parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
parser.add_argument('--dont_train_dict', '-dt', action='store_true',  help='how many epochs to wait train dict -1 means dont train')
parser.add_argument('--test_fruits', action='store_true', help='test on fruit images for bechmark')
parser.add_argument('--add_noise',  action='store_true', help='add noise to input')
parser.add_argument('--inpaint',  action='store_true', help='add noise to input')
parser.add_argument('--grayscale',  action='store_true', help='converte RGB images to grayscale')
parser.add_argument('--test',  action='store_true', help='Run test only')
parser.add_argument('--inpaint_keep_prob', '-p', type=float, default=0.5,
        help='probilty to sample pixel')
parser.add_argument('--recon_loss', default='l2', type=str, choices=['l2', 'l1'])
parser.add_argument('--opt', default='adam', type=str, choices=['adam',
    'momentum'])
parser.add_argument('--clip_val', default=5, type=float, help='max value to clip gradiant')


args = parser.parse_args()


###########################################################
#                  Pre-Proccess Data Sets 
###########################################################
def rgb2gray(X):
    r, g, b = X[..., 0], X[...,1], X[...,2]
    X = (0.2125 * r) + (0.7154 * g) + (0.0721 * b)
    if len(X.shape) == 2:
        X = X[..., np.newaxis]
    return X

def add_noise(X, std):
    noise = np.random.normal(0, std, X.shape)
    return X + noise


def preprocess_data(X, addnoise=False, noise_sigma=0, inpaint=False,
        keep_prob=0.5, whiten=False):
    if whiten:
        X -= np.mean(X, axis=(1, 2), keepdims=True)
        X /= np.std(X, axis=(1, 2), keepdims=True)
    else:
        norm = 255 if 200 < np.max(X) <= 255  else np.max(X) 
        print('norm is {}'.format(norm))
        X = X.astype('float32') / norm
        noise_sigma = float(noise_sigma) / norm
    if addnoise:
        X = add_noise(X, noise_sigma)
 
    if (len(X.shape) == 3 and not args.grayscale) or (len(X.shape) == 2 and args.grayscale):
        X = X[..., np.newaxis]

    if inpaint:
        X *= np.random.choice([0, 1], size=X.shape, p=[1 - keep_prob, keep_prob])
        np.save('inpaint_debug', X)
    return X

###########################################################
#                   Load Data Sets
###########################################################

#
# load data
if args.dataset == 'stl10':  # Data set with unlabeld too large cant loadfully to memory
    if not args.test:
        X_train, X_test= stl10_input.load_data(grayscale=args.grayscale)
        np.random.shuffle(X_train)
    else:
        X_test, _ = stl10_input.load_test(grayscale=args.grayscale)
    # np.random.shuffle(X_test)
elif args.dataset == 'berkeley':
    X_train, X_test = load_berkeley.load(args.grayscale)
    np.random.shuffle(X_train)
elif args.dataset == 'pascal':
    path = load_berkeley.IMGSPATH + 'psacal_gray.npz'
    X_train, X_test = load_berkeley.load(args.grayscale, path)
    np.random.shuffle(X_train)
elif args.dataset == 'city_fruit':
    D = np.load('/data/hillel/data_sets/city_fruit.npz')
    X_train, X_test = D['TRAIN'], D['TEST']
    X_train = np.concatenate((X_train, X_train[:,::-1,...], X_train[...,::-1,:], X_train[:,::-1, ::-1,:]))
    np.random.shuffle(X_train)
else: # dataset is a path of imags to load
    X_test = load_images.load(args.dataset, args.grayscale)
    print(X_test.shape)

noise_sigma = 20


Y_test =  np.empty(shape=X_test.shape)
Y_test[:] = X_test[:]
X_test = preprocess_data(X_test, args.add_noise, noise_sigma, args.inpaint,
        args.inpaint_keep_prob, args.whiten)
Y_test = preprocess_data(Y_test, whiten=args.whiten)
input_shape = X_test.shape[1:]

if not args.test:
    Y_train = np.empty(shape=X_train.shape)
    Y_train[:] = X_train[:]
    X_train = preprocess_data(X_train, args.add_noise, noise_sigma, args.inpaint,
            args.inpaint_keep_prob, args.whiten)
    Y_train = preprocess_data(Y_train, whiten=args.whiten)
    # split train set
    dataset_size = X_train.shape[0]
    train_size = int(np.ceil(dataset_size * 0.9))
    valid_size = dataset_size - train_size
    if not args.dataset == 'berkeley':
        X_valid = X_train[:valid_size]
        X_train = X_train[valid_size:]
        Y_valid = Y_train[:valid_size]
        Y_train = Y_train[valid_size:]
    else:
        X_valid = X_test
        Y_valid = Y_test
    print("training size: {}, test size: {}".format(X_train.shape[0], X_test.shape[0]))
    train_size = X_train.shape[0]
######################################################
#   Plot and save test image
######################################################
def savetstfig(sess, encoder, decoder, inputim, targetim, fname):

    Xhat = decoder.recon_image()
    feed_dict = {encoder.input: inputim}
    if args.inpaint:
        feed_dict[encd_mask] =  (inputim == targetim).astype(float)
    Z, im_hat = sess.run([encoder.output, Xhat], feed_dict)
    
    # im_hat /= np.max(im_hat)
    np.clip(im_hat, 0, 1)  # clip values
    example_ims = DIR_PATH + 'logdir/data/' + fname
    f, axarr = plt.subplots(2, 2)
    np.savez(example_ims, X=inputim, Y=targetim, Z=Z, IM_hat=im_hat)

    cmap = 'gray' #if args.grayscale else 'viridis'
    print('saved example img data de/en at %s' % example_ims)

    axarr[0, 1].axis('off')

    axarr[0,0].imshow(np.squeeze(targetim), cmap=cmap)
    axarr[0,0].set_title('original image')
    axarr[0,0].axis('off')

    axarr[1,1].imshow(np.squeeze(inputim), cmap=cmap)
    axarr[1,1].set_title('noisy image -  psnr: {0:.3} [db]'.format(psnr(targetim, inputim)))
    axarr[1,1].axis('off')

    axarr[1,0].imshow(np.squeeze(im_hat), cmap=cmap)
    axarr[1,0].set_title('reconstructed image -  psnr: {0:.3}\
            [db]'.format(psnr(targetim, im_hat)))
    axarr[1,0].axis('off')

    example_ims = DIR_PATH + 'logdir/plots/' + fname + '.png'
    f.savefig(example_ims)
    plt.close()

    #### TEMP remove ####
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.squeeze(im_hat), cmap=cmap, aspect='normal')
    fig.savefig('result_im', bbox_inches='tight', pad_inches=0)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.squeeze(inputim), cmap=cmap, aspect='normal')
    fig.savefig('noise_im', bbox_inches='tight', pad_inches=0)
    print('saved example img plot de/en at %s' % example_ims)
    scio.savemat('res_mat', {'z': np.squeeze(inputim), 'recon':
        np.squeeze(im_hat), 'y': np.squeeze(targetim)})

def test(encoder, decoder, batch, loss=None):
    test_loss = 0
    recon_loss = 0
    test_iter = 0
    print('batch size {}'.format(args.batch_size))

    for X_batch, Y_batch in test_batch:

        feed_dict = {encoder.input: X_batch, decoder.target: Y_batch}
        if args.inpaint:
            mask = (X_batch == Y_batch).astype(float)
            feed_dict[encd_mask] = mask
        test_loss += sess.run(loss, feed_dict)
        recon_loss += sess.run(l_rec, feed_dict)
        test_iter += 1

    if test_iter:
        test_loss, recon_loss = test_loss/test_iter, recon_loss/test_iter

    # for debug save We/Wd
    We_Wd = DIR_PATH + 'logdir/data/We_Wd'
    We, Wd, theta = sess.run([encoder._We, encoder._Wd, encoder._theta])
    np.savez(We_Wd, Wd=Wd, We=We, theta=theta)

    # plot example image
    for ex_i in range(min(20, len(X_test))):
        # i = np.random.randint(X_test.shape[0], size=1)
        x = X_test[[ex_i], :]
        y = Y_test[[ex_i], :]
        savetstfig(sess, encoder, decoder, x, y, 'example'+str(ex_i))

    return test_loss, recon_loss



#######################################################
#   Log-network for tensorboard
#######################################################
tensorboard_path = '/data/hillel/tb/' + args.name + '/'
if not os.path.isdir(tensorboard_path):
    os.mkdir(tensorboard_path)

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
encoder, decoder, encd_mask = sparse_aed.build_model(args, input_shape)


#
# LOSS

# l2-loss with index -1 is reconstruction of final encoding 
# l2-loss with index 0 is of indermidate encode
l_rec  = (1.0 / (input_shape[0]*input_shape[1])) * \
            decoder.recon_loss_layer_i(-1, encd_mask, args.recon_loss, 'recon_loss')
         #args.middle_loss_factor * \
         #decoder.recon_loss_layer_i(mdl_i, args.recon_loss, 'recon_loss_stage_%d'%mdl_i)
loss = args.recon_factor * l_rec
half_ker = args.kernel_size // 2
if args.total_variation:
    loss_total_var =  tf.reduce_mean(tf.image.total_variation(decoder.recon_image()))
    loss += args.total_variation * loss_total_var
if args.ms_ssim:
    _ms_ssim = ms_ssim.tf_ms_ssim(
            decoder.target[:,half_ker:-half_ker,half_ker:-half_ker,...],
            decoder.recon_image()[:,half_ker:-half_ker,half_ker:-half_ker,...], level=4
            )
    loss_ms_ssim = (1 - _ms_ssim)
    loss +=  args.ms_ssim * loss_ms_ssim
if args.sparse_factor:
    l_sparse = tf.reduce_mean(tf.abs(encoder.output), name='l1')
    loss += args.sparse_factor * l_sparse

#######################################################
# Add vars to summary
#######################################################
with tf.name_scope('encoder'):
    with tf.name_scope('We'):
        variable_summaries(encoder._We)
    with tf.name_scope('Wd'):
        variable_summaries(encoder._Wd)
    with tf.name_scope('threshold'):
        variable_summaries(encoder._theta)
with tf.name_scope('decoder'):
    variable_summaries(decoder.decoder)
with tf.name_scope('sparse_code'):
    variable_summaries(encoder.output)
tf.summary.scalar('encoded_sparsity',
        tf.reduce_mean(tf.count_nonzero(encoder.output, 1)))
if not args.test:
    if args.sparse_factor:
        tf.summary.scalar('l1_loss', l_sparse)
    tf.summary.scalar('recon_loss', l_rec)
    if args.total_variation:
        tf.summary.scalar('smooth', loss_total_var)
    tf.summary.scalar('ms_ssim', loss_ms_ssim)
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
def lerning_rate_timedecay(learning_rate, decay_rate, global_step, decay_steps):
    learning_rate = tf.train.inverse_time_decay(
            learning_rate=args.learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
            )
    return learning_rate

def clip_grad(optimizer, loss,  val, tr_vars, global_step):
    gvs = optimizer.compute_gradients(loss, tr_vars)
    capped_gvs = [(tf.clip_by_value(grad, -val, val), var) for grad, var in gvs]
    optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    return optimizer

def get_opt(name, lr=0.001, momentum=0.9):
    if name == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif name == 'momentum':
        return  tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)

if not args.test:
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder/")
    global_step = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name='global_step'
            )
    lr_rate = lerning_rate_timedecay(
            args.learning_rate,
            30,
            global_step,
            train_size // args.batch_size
            )
    learning_rate_var = tf.Variable(args.learning_rate)
    optimizer = get_opt(args.opt, learning_rate_var).minimize(loss, global_step=global_step)
    # optimizer_en = get_opt(args.opt, learning_rate_var).minimize(loss, var_list=encoder_vars, global_step=global_step_en)
    # optimizer_en = clip_grad(optimizer_en, loss, args.clip_val, encoder_vars, global_step_en)

    #if not args.dont_train_dict:
    #    global_step_de = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_de')
    #    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder/")
    #    lr_rate = lerning_rate_timedecay(1.5 * args.learning_rate, 30, global_step_de,
    #         train_size // args.batch_size)
    #    optimizer_de =  get_opt(args.opt, 1.5 * learning_rate_var).minimize(loss,
    #            var_list=decoder_vars, global_step=global_step_de)

       # optimizer_de = clip_grad(optimizer_de, loss, args.clip_val, encoder_vars,
       #         global_step_de)


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
            # reshuffle batch
            idx = np.random.permutation(X.shape[0])
            X = X[idx,...]
            Y = Y[idx,...]
        yield batch_X, batch_Y
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
    valid_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name +
            '_valid')
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

    if args.test:
        test_batch = nextbatch(X=X_test, Y=Y_test, batch_size=5, run_once=True)
        test_loss, test_recon_loss = test(encoder, decoder, test_batch, loss)
        print('test loss: %f recon loss: %f' % (test_loss, test_recon_loss))
        exit(0)

    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("all_zero", all_zero)

    print('number of epochs %d'%args.num_epochs)
    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0
        print('epoch number:%d', epoch)
        train_batch = nextbatch(X=X_train, Y=Y_train, batch_size=args.batch_size,
                                run_once=True)
        b_num = 0
        for X_batch, Y_batch in train_batch:
            b_num += 1
            feed_dict = {encoder.input: X_batch, decoder.target: Y_batch}
            if args.inpaint:
                mask = (X_batch == Y_batch).astype(float)
                feed_dict[encd_mask] = mask
            # np.savez('debug', X=X_batch, Y=Y_batch, mask=mask)
            _, iter_loss = sess.run([optimizer, loss], feed_dict)
            #if not  args.dont_train_dict:
            #    _, iter_loss = sess.run([optimizer_de, loss], feed_dict)
            train_loss.append(iter_loss)
            epoch_loss += iter_loss


            if b_num % 30 == 0:
                summary = sess.run(merged, feed_dict)
                train_summ_writer.add_summary(summary, global_step=global_step.eval(session=sess))

            if b_num % 30 == 0:
                valid_loss = 0
                v_itr = 0
                sp_in_itr = 0
                sp_out_itr = 0
                l1 = 0
                recon = 0
                vaild_batch = nextbatch(X=X_valid, Y=Y_valid, batch_size=5, run_once=True)
                for Xv_batch, Yv_batch in vaild_batch:
                    v_itr += 1
                    feed_dict = {encoder.input: Xv_batch, decoder.target: Yv_batch}
                    if args.inpaint:
                        mask = (Xv_batch == Yv_batch).astype(float)
                        feed_dict[encd_mask] = mask

                    iter_loss, iter_recon, enc_out, summary  = \
                            sess.run([loss, l_rec, encoder.output, merged], feed_dict)
                    valid_summ_writer.add_summary(summary,
                            global_step=global_step.eval(session=sess))
                    sp_out_itr += np.count_nonzero(enc_out)/enc_out.shape[0]
                    sp_in_itr += np.count_nonzero(X_batch)/Xv_batch.shape[0]
                    valid_loss += iter_loss
                    recon += iter_recon
                valid_loss /= v_itr
                recon /= v_itr
                validation_loss.append(valid_loss)
                valid_sparsity_out.append(sp_out_itr/v_itr)
                valid_sparsity_in.append(sp_in_itr/v_itr)

                if valid_loss <= np.min(validation_loss):
                    if args.save_model:
                        f_name = MODEL_DIR + 'csc_u{}_'.format(args.unroll_count)
                        saver.save(sess, f_name, global_step=global_step)
                        print('saving model at: %s'%f_name) 
                if len(validation_loss)  > 5:
                    if (valid_loss > validation_loss[-2]).all():
                        learning_rate_var *= 0.9
                        print('loading model %s'%MODEL_DIR)
                        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
                        print('decreasing learning_rate to\
                                {}'.format(learning_rate_var.eval()))
                print('valid loss: %f recon loss: %f encoded sparsity: %f' %
                    (valid_loss, recon, valid_sparsity_out[-1]))
        print('epoch %d: loss val:%f' % (epoch, args.batch_size  * epoch_loss / X_train.shape[0]))

    print('='*40)
    decoder_filters = decoder.decoder.eval()
    decoder_filters_path = DIR_PATH + 'logdir/data/decoder_filters'
    np.save(decoder_filters_path, decoder_filters)
    print('saved decoder filters at path: %s' % decoder_filters_path)

    test_batch = nextbatch(X=X_test, Y=Y_test, batch_size=5, run_once=True)
    test_loss, test_recon_loss = test(encoder, decoder, test_batch, loss)
    if args.test_fruits:
        X_fruit = load_images.load_fruit(args.grayscale)
        Y_fruit = np.empty(shape=X_fruit.shape)
        Y_fruit[:] = X_fruit[:]
        X_fruit = preprocess_data(X_fruit, args.add_noise, noise_sigma, args.inpaint, args.inpaint_keep_prob, args.whiten)

        Y_fruit = preprocess_data(Y_fruit, whiten=args.whiten)

        for i, (xfruit, yfruit)  in enumerate(zip(X_fruit, Y_fruit)):
            savetstfig(sess, encoder, decoder, xfruit[np.newaxis, :], yfruit[np.newaxis, :], 'fruit'+str(i))


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
    cmap = 'gray' if len(f.shape) == 2 else 'classic'
    plt.imshow(f.T, interpolation='bilinear', cmap=cmap)
    fid += 1
plt.savefig(DIR_PATH + 'logdir/plots/filters.png')
print('seved plot of dict filter atoms in {}'.format(DIR_PATH + 'logdir/plots/filters.png'))

print('='*40)
print('test loss: %f recon loss: %f' % (test_loss, test_recon_loss))

