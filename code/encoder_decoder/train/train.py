import os
import sys
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.python import debug as tf_debug
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(os.path.abspath(DIR_PATH + '../../'))

from approx_sae.approx_conv2d_sparse_ae import ApproxCSC
from approx_sae import approx_sae_losses
from Utils import data_handler
from Utils import ms_ssim
from Utils.psnr import psnr as psnr

######################################################
#   Plot and save test image
######################################################
def savetstfig(sess, encoder, decoder, inputim, targetim, fname):

    Xhat = decoder.output
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

def get_opt(lr=0.001, momentum=0.9):
    opt_name =HYPR_PARAMS.get('optimizer', 'adam')
    if  opt_name == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif opt_name == 'momentum':
        return  tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)


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

def config_train_tb(_encoder, _decoder):
    tensorboard_path = '/data/hillel/tb/' + HYPR_PARAMS['name'] + '/'
    if not os.path.isdir(tensorboard_path):
        os.mkdir(tensorboard_path)

    with tf.name_scope('encoder'):
        with tf.name_scope('We'):
            variable_summaries(_encoder._We)
        with tf.name_scope('Wd'):
            variable_summaries(_encoder._Wd)
        with tf.name_scope('threshold'):
            variable_summaries(_encoder._theta)
    with tf.name_scope('decoder'):
        variable_summaries(_decoder.convdict)
    with tf.name_scope('sparse_code'):
        variable_summaries(_encoder.output)
    tf.summary.scalar('encoded_sparsity', 
        tf.reduce_mean(tf.count_nonzero(_encoder.output, axis=[1,2,3])))
    tf.summary.image('input', _encoder.input)
    tf.summary.image('output', _decoder.output)
    tf.summary.image('target', _decoder.target)
    return tensorboard_path

def reconstruction_loss(_model):
    dist = 0
    ms_ssim = 0

    out = _model.decoder.output
    target = _model.decoder.target
    boundry = (_model.encoder.kernel_size // 2,_model.encoder.kernel_size // 2)

    if HYPR_PARAMS['disttyp'] == 'l1':
        dist_loss = approx_sae_losses.l1(_out=_out, _target=_target, _boundry=boundry)
        tf.summary.scalar('dist_l1', dist_loss, collections='TB_LOSS')
    elif HYPR_PARAMS['disttyp'] == 'smooth_l1':
       dist_loss = approx_sae_losses.smooth_l1(_out=_out, _target=_target, _boundry=boundry)
       tf.summary.scalar('dist_smoothl1', dist_loss, collections='TB_LOSS')
    elif HYPR_PARAMS['disttyp'] == 'l2':
       dist_loss = approx_sae_losses.l2(_out=out, _target=target, _boundry=boundry)
       tf.summary.scalar('dist_l2', dist_loss, collections='TB_LOSS')

    if HYPR_PARAMS['ms_ssim']:
        ms_ssim = ms_ssim.tf_ms_ssim(
                target[:,half_ker:-half_ker,half_ker:-half_ker,...],
                out[:,half_ker:-half_ker,half_ker:-half_ker,...]
        )
        tf.summary.scalar('ms_ssim', ms_ssim, collections='TB_LOSS')
    return dist, (1 - ms_ssim)

def sparsecode_loss(_model):
    sparse_loss = 0
    similarity_loss = 0

    if HYPR_PARAMS['sparse_factor']:
        sparse_loss = approx_sae_losses.l1(_model.output, tf.zeros_like(_model.output))
        tf.summary.scalar('l1_sparse', sparse_loss, collections='TB_LOSS')
    if HYPR_PARAMS['sparse_sim_factor']:
        similarity_loss = approx_sae_losses.sc_similarity(
            _x=_model.sparsecode,
            _chunk_size=HYPR_PARAMS['dup_count'],
            _chunk_count=HYPR_PARAMS['batch_size']
        )
        tf.summary.scalar('sc_sim', similarity_loss, collections='TB_LOSS')
    return sparse_loss, similarity_loss
    
class Saver():
    """Help handle save/restore logic"""
    def __init__(self, **kwargs):
        self.save = kwargs.get('save_model', False)
        self.load = kwargs.get('load_model', False)
        self.path = kwargs['dir_path'] + kwargs['name'] + '/'
        self.saver = None

    def  __call__(self): 
        if self.save or self.load:
            MODEL_DIR = DIR_PATH + '/logdir/models/' + self.name + '/'
            self.saver = tf.train.Saver(max_to_keep=1)
            if not os.path.isdir(MODEL_DIR):
                os.mkdir(MODEL_DIR)

    def maybe_load(self, load_name):
        if self.load:
            if load_name != '':
                LOAD_DIR = DIR_PATH + '/logdir/models/' + load_name + '/'
            else:
                LOAD_DIR = self.path
            if os.listdir(LOAD_DIR):
                print('loading model')
                saver.restore(sess, tf.train.latest_checkpoint(LOAD_DIR))
                return True
            else:
                print('no cpk to load running with random init')
                return False

    @property
    def saver(self):
        return self.saver


       
def train(_model, _datahandler):

    tensorboard_path = config_train_tb(_model.encoder, _model.decoder)

    dist_loss, msssim_loss = reconstruction_loss(_model)
    _reconstructoin_loss =\
        HYPR_PARAMS['recon_factor'] * dist_loss  + \
        HYPR_PARAMS['ms_ssim'] * msssim_loss

    sparse_loss, similarity_loss = sparsecode_loss(_model)
    _sparsecode_loss = \
        HYPR_PARAMS['sparse_factor'] * sparse_loss + \
        HYPR_PARAMS['sparse_sim_factor'] * similarity_loss

    loss = _reconstructoin_loss + _sparsecode_loss
    saver_mngr = Saver(dir_path=DIR_PATH, **HYPR_PARAMS)
    saver_mngr()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate_var = tf.Variable(args.learning_rate)
    optimizer = get_opt(learning_rate_var).minimize(loss, global_step=global_step)

    train_dh = _datahandler.train_gen(HYPR_PARAMS['batch_size'])

    ###################################################################
    #                Training   +   Results
    ###################################################################
    train_loss = []
    validation_loss = []
    valid_sparsity_out = []
    valid_sparsity_in = []
    epoch_loss = 0

    with tf.Session() as sess:

        tf.global_variables_initializer().run(session=sess)
        print('Initialized')

        merged_only_tr = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        merged = tf.summary.merge_all(key='TB_LOSS')
        train_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name)
        train_summ_writer.add_graph(sess.graph)
        valid_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name +
                '_valid')

        saver_mngr.maybe_load(HYPR_PARAMS['load_name'])

        if HYPR_PARAMS.get('test', False):  # TODO: fix this
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
            b_num = 0

            for X_batch, Y_batch in train_dh:
                b_num += 1
                feed_dict = {_model.input: X_batch, _model.target: Y_batch}
                if inpaint:
                    mask = (X_batch == Y_batch).astype(float)
                    feed_dict[encd_mask] = mask
                _, iter_loss = sess.run([optimizer, loss], feed_dict)
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
                    for Xv_batch, Yv_batch in valid_dh:
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
        decoder_filters = decoder.convdict.eval()
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

def main():
    dh = data_handler.DataHandlerBase.factory(norm_val=255, **HYPR_PARAMS)
    model  = ApproxCSC()
    model.build_model(
        unroll_count=args.unroll_count,
        L=1, batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        shrinkge_type=args.shrinkge_type,
        kernel_count=args.kernel_count,
        channel_size=dh.shape[-1]
    )
    train(_datahandler=dh, _model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse encoder decoder model')
    parser.add_argument('-b', '--batch_size', default=2,
                                type=int, help='size of train batches')
    parser.add_argument('-n', '--num_epochs', default=1, type=int,
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
    parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model', dest='load_model', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--name', default='lista_conv2d', type=str, help='used for\
            creating load/store log dir names')
    parser.add_argument('--load_name', default='', type=str, help='used to\
            load from a model with "name" diffrent from this model name')
    parser.add_argument('--dataset', default='berkeley', choices=['mnist','stl10', 'cifar10', 'pascal'])
    parser.add_argument('--sparse_factor', '-sf',  default=0.5, type=float)
    parser.add_argument('--sparse_sim_factor',  default=0.0, type=float)
    parser.add_argument('--recon_factor', '-rf',  default=1.0, type=float)
    parser.add_argument('--ms_ssim', '-ms',  default=0.0, type=float)
    parser.add_argument('--dup_count', '-dc',  default=0, type=int)
    parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
    parser.add_argument('--dont_train_dict', action='store_true',  help='how many epochs to wait train dict -1 means dont train')
    parser.add_argument('--task',  default='denoise', choices=['deniose', 'inpaint'], 
            help='add noise to input')
    parser.add_argument('--grayscale',  action='store_true', help='converte RGB images to grayscale')
    parser.add_argument('--inpaint_keep_prob', '-p', type=float, default=0.5,
            help='probilty to sample pixel')
    parser.add_argument('--noise_sigma', '-ns', type=float, default=20,
            help='probilty to sample pixel')
    parser.add_argument('--disttyp', '-dt', default='l2', type=str, choices=['l2', 'l1', 'smoothl1'])
    args = parser.parse_args()

    global HYPR_PARAMS
    HYPR_PARAMS = vars(args)
    main()
