import os
import sys
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.python import debug as tf_debug
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
from Utils import tf_train_utils
from evaluate import evaluate

#######################################################
#   Training Procedure                                #
#######################################################
def train(_model, _datahandler):

    dist_loss, msssim_loss = \
        approx_sae_losses.reconstruction_loss(_model, HYPR_PARAMS['disttyp'], HYPR_PARAMS['ms_ssim'])# reconstruction_loss(_model)
    _reconstructoin_loss =\
        HYPR_PARAMS['recon_factor'] * dist_loss  + \
        HYPR_PARAMS['ms_ssim'] * msssim_loss
    loss = _reconstructoin_loss 

    if HYPR_PARAMS['sparse_factor']:
        sparse_loss, _ = approx_sae_losses.sparsecode_loss(_model)
        _sparsecode_loss = \
            HYPR_PARAMS['sparse_factor'] * sparse_loss
            # + HYPR_PARAMS['sparse_sim_factor'] * similarity_loss
        loss += _sparsecode_loss


    tensorboard_path = tf_train_utils.config_train_tb(_model, HYPR_PARAMS['name'], loss=loss)

    saver_mngr = tf_train_utils.Saver(dir_path=DIR_PATH, **HYPR_PARAMS)
    saver_mngr()

    opt_name = HYPR_PARAMS.get('optimizer', 'adam')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate_var = tf.Variable(args.learning_rate)
    optimizer = tf_train_utils.get_optimizer(opt_name, learning_rate_var).minimize(loss, global_step=global_step)

    batch_size = HYPR_PARAMS['batch_size']
    if HYPR_PARAMS['task'] == 'multi_denoise':
        batch_size *= HYPR_PARAMS['dup_count']

    train_dh = _datahandler.train_gen(batch_size)
    valid_dh = _datahandler.valid_gen(batch_size)
    test_dh = _datahandler.test_gen(10)
    print('VALID BATCHED {}'.format(valid_dh._batchs_per_epoch))
    ###################################################################
    #                Training   +   Results
    ###################################################################
    validation_loss = []
    valid_sparsity_out = []

    with tf.Session() as sess:

        tf.global_variables_initializer().run(session=sess)
        print('Initialized')

        np.savez('dbugfile_untied', Wd=sess.run(_model.encoder._Wd),
                We=sess.run(_model.encoder._We))

        merged_only_tr = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        merged = tf.summary.merge_all(key='TB_LOSS')
        train_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name)
        train_summ_writer.add_graph(sess.graph)
        valid_summ_writer = tf.summary.FileWriter(tensorboard_path + args.name + '_valid')

        saver_mngr.maybe_load(HYPR_PARAMS['load_name'], sess)

        if HYPR_PARAMS.get('test', False):  # TODO: fix this
            test_batch = nextbatch(X=X_test, Y=Y_test, batch_size=5, run_once=True)
            test_loss, test_recon_loss = test(encoder, decoder, test_batch, loss)
            print('test loss: %f recon loss: %f' % (test_loss, test_recon_loss))
            exit(0)

        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        print('number of epochs %d'%args.num_epochs)
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss = 0
            print('epoch number:%d'%epoch)

            for X_batch, Y_batch in train_dh:

                iter_loss = 0
                feed_dict = {_model.input: X_batch, _model.target: Y_batch}
                if HYPR_PARAMS['task'] == 'inpaint':
                    mask = (X_batch == Y_batch).astype(float)
                    feed_dict[encd_mask] = mask
                _, iter_loss = sess.run([optimizer, loss], feed_dict)
                epoch_loss += iter_loss

                if train_dh.batch_num % 30 == 0:
                    summary = sess.run([merged, merged_only_tr], feed_dict)
                    for s in summary:
                        train_summ_writer.add_summary(s, global_step=global_step.eval(session=sess))

                if train_dh.batch_num % 50  == 0:
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
                            print('valid loss {} \n all validation loss {}'.format(valid_loss, validation_loss))
                            tf_train_utils.change_lr_val(sess, learning_rate_var, 0.9) 
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

def param_count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def main():
    dh = data_handler.DataHandlerBase.factory(norm_val=255, **HYPR_PARAMS)
    model = ApproxCSC(type=HYPR_PARAMS['model_type'])
    model.build_model(
        amount_stacked=HYPR_PARAMS['amount_stacked'],
        unroll_count=HYPR_PARAMS['unroll_count'],
        L=1, batch_size=HYPR_PARAMS['batch_size'],
        kernel_size=HYPR_PARAMS['kernel_size'],
        shrinkge_type=HYPR_PARAMS['shrinkge_type'],
        kernel_count=HYPR_PARAMS['kernel_count'],
        channel_size=dh.shape[-1],
        norm_kernal=HYPR_PARAMS['norm_kernal']
    )
    print('Amount of params: {}'.format(param_count()))
    train(_datahandler=dh, _model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse encoder decoder model')
    parser.add_argument('-b', '--batch_size', default=10,
                                type=int, help='size of train batches')
    parser.add_argument('-n', '--num_epochs', default=1, type=int,
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
    parser.add_argument('--dup_count', '-dc',  default=1, type=int)
    parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
    parser.add_argument('--dont_train_dict', action='store_true',  help='how many epochs to wait train dict 0 means dont train')
    parser.add_argument('--task',  default='denoise_dynamicthrsh', choices=['denoise', 'denoise_dynamicthrsh', 'inpaint', 'multi_noise'], 
            help='add noise to input')
    parser.add_argument('--grayscale',  action='store_true', help='converte RGB images to grayscale')
    parser.add_argument('--inpaint_keep_prob', '-p', type=float, default=0.5,
            help='probilty to sample pixel')
    parser.add_argument('--noise_sigma', '-ns', type=float, default=20,
            help='noise magnitude')
    parser.add_argument('--disttyp', '-dt', default='l1', type=str, choices=['l2', 'l1', 'smoothl1'])
    parser.add_argument('--model_type', '-mt', default='dynamicthrsh_untied', choices=['convdict', 'convmultidict', 'untied', 'dynamicthrsh', 'dynamicthrsh_untied'])
    parser.add_argument('--norm_kernal',  action='store_true', help='keep kernals with unit kernels')
    parser.add_argument('--amount_stacked',  default=1, type=int,
    help='Amount of LISTA AE to stack')
#TODO: add args for dynamic thresholding
    args = parser.parse_args()

    global HYPR_PARAMS
    HYPR_PARAMS = vars(args)
    main()
