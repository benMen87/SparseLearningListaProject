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

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(os.path.abspath(DIR_PATH))
sys.path.append(os.path.abspath(DIR_PATH) + '/../')

from approx_sae.approx_conv2d_sparse_ae import ApproxCSC
from approx_sae import approx_sae_losses
from Utils import data_handler
from Utils import ms_ssim
from evaluate import evaluate

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
    dist_loss = 0
    _ms_ssim = 0

    out = _model.decoder.output
    target = _model.decoder.target
    boundry = (_model.encoder.kernel_size // 2,_model.encoder.kernel_size // 2)

    if HYPR_PARAMS['disttyp'] == 'l1':
        dist_loss = approx_sae_losses.l1(_out=out, _target=target, _boundry=boundry)
        tf.summary.scalar('dist_l1', dist_loss, collections=['TB_LOSS'])
    elif HYPR_PARAMS['disttyp'] == 'smooth_l1':
        dist_loss = approx_sae_losses.smooth_l1(_out=_out, _target=_target, _boundry=boundry)
        tf.summary.scalar('dist_smoothl1', dist_loss, collections=['TB_LOSS'])
    elif HYPR_PARAMS['disttyp'] == 'l2':
        dist_loss = approx_sae_losses.l2(_out=out, _target=target, _boundry=boundry)
        tf.summary.scalar('dist_l2', dist_loss, collections=['TB_LOSS'])

    if HYPR_PARAMS['ms_ssim']:
        half_ker = _model.encoder.kernel_size // 2
        _ms_ssim = ms_ssim.tf_ms_ssim(
            target[:,half_ker:-half_ker,half_ker:-half_ker,...],
            out[:,half_ker:-half_ker,half_ker:-half_ker,...],
            level=4
        )
        tf.summary.scalar('ms_ssim', _ms_ssim, collections=['TB_LOSS'])
    return dist_loss, (1 - _ms_ssim)

def sparsecode_loss(_model):
    sparse_loss = 0
    similarity_loss = 0

    if HYPR_PARAMS['sparse_factor']:
        sparse_loss = tf.reduce_mean(tf.abs(_model.sparsecode))
        tf.summary.scalar('l1_sparse', sparse_loss, collections=['TB_LOSS'])
    if HYPR_PARAMS['sparse_sim_factor']:
        similarity_loss = approx_sae_losses.sc_similarity(
            _x=_model.sparsecode,
            _chunk_size=HYPR_PARAMS['dup_count'],
            _chunk_count=HYPR_PARAMS['batch_size']
        )
        tf.summary.scalar('sc_sim', similarity_loss, collections=['TB_LOSS'])
    return sparse_loss, similarity_loss
    
class Saver():
    """Help handle save/restore logic"""
    def __init__(self, **kwargs):
        self._save = kwargs.get('save_model', False)
        self._load = kwargs.get('load_model', False)
        self._name = kwargs['name']
        self._path = DIR_PATH + '/logdir/models/' + self._name + '/'
        self._saver = None

    def  __call__(self): 
        if self._save or self._load:
            self._saver = tf.train.Saver(max_to_keep=1)
            if not os.path.isdir(self._path):
                os.mkdir(self._path)

    def maybe_load(self, load_name, sess):
        if self._load:
            if load_name != '':
                LOAD_DIR = DIR_PATH + '/logdir/models/' + load_name + '/'
            else:
                LOAD_DIR = self._path
            if os.listdir(LOAD_DIR):
                print('loading model')
                self._saver.restore(sess, tf.train.latest_checkpoint(LOAD_DIR))
                return True
            else:
                print('no cpk to load running with random init')
                return False

    def save(self, sess, global_step):
        self._saver.save(sess, self._path, global_step=global_step)

    def restore(self, sess):
        self._saver.restore(sess, tf.train.latest_checkpoint(self._path))

    @property
    def saver(self):
        return self._saver
       
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

    batch_size = HYPR_PARAMS['batch_size']
    if HYPR_PARAMS['task'] == 'multi_denoise':
        batch_size *= HYPR_PARAMS['dup_count']

    train_dh = _datahandler.train_gen(batch_size)
    valid_dh = _datahandler.valid_gen(batch_size)
    test_dh = _datahandler.test_gen(10)

    ###################################################################
    #                Training   +   Results
    ###################################################################
    validation_loss = []
    valid_sparsity_out = []
    with tf.Session() as sess:

        tf.global_variables_initializer().run(session=sess)
        print('Initialized')

        merged_only_tr = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        merged = tf.summary.merge_all(key='TB_LOSS')
        train_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name)
        train_summ_writer.add_graph(sess.graph)
        valid_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name +
                '_valid')

        saver_mngr.maybe_load(HYPR_PARAMS['load_name'], sess)

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
                if HYPR_PARAMS['task'] == 'inpaint':
                    mask = (X_batch == Y_batch).astype(float)
                    feed_dict[encd_mask] = mask
                _, iter_loss = sess.run([optimizer, loss], feed_dict)
                epoch_loss += iter_loss

                if b_num % 30 == 0:
                    summary = sess.run([merged, merged_only_tr], feed_dict)
                    for s in summary:
                        train_summ_writer.add_summary(s, global_step=global_step.eval(session=sess))

                if b_num % 30 == 0:
                    valid_loss = 0
                    v_itr = 0
                    sp_out_itr = 0
                    l1 = 0
                    recon = 0

                    for Xv_batch, Yv_batch in valid_dh:
                        v_itr += 1
                        feed_dict = {_model.input: Xv_batch, _model.target: Yv_batch}
                        if HYPR_PARAMS['task'] == 'inpaint':
                            mask = (Xv_batch == Yv_batch).astype(float)
                            feed_dict[encd_mask] = mask
                        iter_loss, iter_recon, enc_out, summary  = \
                                sess.run([loss, _reconstructoin_loss, _model.sparsecode, merged], feed_dict)
                        valid_summ_writer.add_summary(summary, global_step=global_step.eval(session=sess))
                        sp_out_itr += np.count_nonzero(enc_out)/enc_out.shape[0]
                        valid_loss += iter_loss
                        recon += iter_recon
                    valid_loss /= v_itr
                    recon /= v_itr

                    validation_loss.append(valid_loss)
                    valid_sparsity_out = sp_out_itr/v_itr

                    if valid_loss <= np.min(validation_loss):
                        if args.save_model:
                            saver_mngr.save(sess, global_step=global_step)
                            print('saving model at: %s'%saver_mngr._name) 
                    if len(validation_loss)  > 5:
                        if (valid_loss > validation_loss[-5:]).all():
                            learning_rate_var *= 0.9
                            saver_mngr.restore(sess)
                            print('decreasing learning_rate to\
                                    {}'.format(learning_rate_var.eval()))
                    print('valid loss: %f recon loss: %f encoded sparsity: %f' %
                        (valid_loss, recon, valid_sparsity_out))
            print('epoch %d: loss val:%f' % (epoch, args.batch_size  * epoch_loss / X_train.shape[0]))

        #run test
        save_path = DIR_PATH + '/logdir/'
        test_loss = evaluate.test(sess, _model, test_dh, loss, save_path)
        decoder_filters = np.squeeze(_model.decoder.convdict.eval(session=sess))
    
    print('='*40 + '\nTEST LOSS {}\n'.format(test_loss) + '='*40)
    plt.figure()
    plot_num = np.sqrt(args.kernel_count)
    fid = 0
    for f in decoder_filters.T:
        plt.subplot(plot_num, plot_num, fid+1)
        cmap = 'gray' if len(f.shape) == 2 else 'classic'
        plt.imshow(f.T, interpolation='bilinear', cmap=cmap)
        fid += 1
    plt.savefig(DIR_PATH + 'logdir/plots/filters.png')
    print('seved plot of dict filter atoms in {}'.format(DIR_PATH + 'logdir/plots/filters.png'))

def main():
    dh = data_handler.DataHandlerBase.factory(norm_val=255, **HYPR_PARAMS)
    model  = ApproxCSC()
    model.build_model(
        unroll_count=HYPR_PARAMS['unroll_count'],
        L=1, batch_size=HYPR_PARAMS['batch_size'],
        kernel_size=HYPR_PARAMS['kernel_size'],
        shrinkge_type=HYPR_PARAMS['shrinkge_type'],
        kernel_count=HYPR_PARAMS['kernel_count'],
        channel_size=dh.shape[-1]
    )
    train(_datahandler=dh, _model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse encoder decoder model')
    parser.add_argument('-b', '--batch_size', default=10,
                                type=int, help='size of train batches')
    parser.add_argument('-n', '--num_epochs', default=0, type=int,
                                help='number of epochs steps')
    parser.add_argument('-ks', '--kernel_size', default=7, type=int,
                                help='kernel size to be used in lista_conv')
    parser.add_argument('-kc', '--kernel_count', default=64, type=int,
                                help='amount of kernel to use in lista_conv')
    parser.add_argument('--dilate', '-dl', action='store_true')
    parser.add_argument('-u', '--unroll_count', default=3,
         type=int, help='Amount of Reccurent timesteps for decoder')
    parser.add_argument('--shrinkge_type', default='soft thresh',
                            choices=['soft thresh', 'smooth soft thresh'])
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model', dest='load_model', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--name', default='lista_conv2d', type=str, help='used for\
            creating load/store log dir names')
    parser.add_argument('--load_name', default='', type=str, help='used to\
            load from a model with "name" diffrent from this model name')
    parser.add_argument('--dataset', default='pascal_small', choices=['mnist','stl10', 'cifar10', 'pascal', 'pascal_small'])
    parser.add_argument('--sparse_factor', '-sf',  default=0.5, type=float)
    parser.add_argument('--sparse_sim_factor',  default=0, type=float)
    parser.add_argument('--recon_factor', '-rf',  default=1.0, type=float)
    parser.add_argument('--ms_ssim', '-ms',  default=0.0, type=float)
    parser.add_argument('--dup_count', '-dc',  default=2, type=int)
    parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
    parser.add_argument('--dont_train_dict', action='store_true',  help='how many epochs to wait train dict 0 means dont train')
    parser.add_argument('--task',  default='multi_denoise', choices=['denoise', 'inpaint', 'multi_denoise'], 
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
