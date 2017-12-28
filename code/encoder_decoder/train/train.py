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
from Utils import lista_args
from evaluate import evaluate

#######################################################
#   Training Procedure 
#######################################################
def train(_model, _datahandler):
    
    dist_loss, msssim_loss  =\
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

    sparsity_counter_opt = tf_train_utils.tf_sparse_count(_model.sparsecode)

    tensorboard_path = tf_train_utils.config_train_tb(_model, HYPR_PARAMS['name'], loss=loss)

    saver_mngr = tf_train_utils.Saver(dir_path=DIR_PATH, **HYPR_PARAMS)
    saver_mngr()

    opt_name = HYPR_PARAMS.get('optimizer', 'adam')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate_var = tf.Variable(HYPR_PARAMS['learning_rate'])
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
        train_summ_writer = tf.summary.FileWriter(tensorboard_path +
                HYPR_PARAMS['name'])
        train_summ_writer.add_graph(sess.graph)
        valid_summ_writer = tf.summary.FileWriter(tensorboard_path +
                HYPR_PARAMS['name'] + '_valid')

        saver_mngr.maybe_load(HYPR_PARAMS['load_name'], sess)

        if HYPR_PARAMS.get('test', False):  # TODO: fix this
            test_batch = nextbatch(X=X_test, Y=Y_test, batch_size=5, run_once=True)
            test_loss, test_recon_loss = test(encoder, decoder, test_batch, loss)
            print('test loss: %f recon loss: %f' % (test_loss, test_recon_loss))
            exit(0)

        if HYPR_PARAMS['debug']:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        print('number of epochs %d'%HYPR_PARAMS['num_epochs'])
        for epoch in range(1, HYPR_PARAMS['num_epochs'] + 1):
            epoch_loss = 0
            print('epoch number:%d'%epoch)

            for X_batch, Y_batch in train_dh:
                iter_loss = 0
                feed_dict = {_model.input: X_batch, _model.target: Y_batch}
                #
                # Creat binary mask for inpiant task
                if HYPR_PARAMS['task'] == 'inpaint':
                    mask = (X_batch == Y_batch).astype(float)
                    feed_dict[encd_mask] = mask
                #
                # Lets log stats prior to optimizer run --> clearer view of network initial state
                if (train_dh.batch_num-1) % 30 == 0:
                    summary = sess.run([merged, merged_only_tr], feed_dict)
                    for s in summary:
                        train_summ_writer.add_summary(s, global_step=global_step.eval(session=sess))
                #
                # This is the actual bp optimization
                _, iter_loss, = sess.run([optimizer, loss], feed_dict)
                epoch_loss += iter_loss
                
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
                        iter_loss, iter_recon, sc_cnt  = \
                                sess.run([loss, _reconstructoin_loss, sparsity_counter_opt], feed_dict)
                        sp_out_itr += sc_cnt
                        valid_loss += iter_loss
                        recon += iter_recon
                    valid_loss /= v_itr
                    recon /= v_itr

                    validation_loss.append(valid_loss)
                    valid_sparsity_out = sp_out_itr/v_itr
                    if valid_loss <= np.min(validation_loss):
                        if HYPR_PARAMS['save_model']:
                            saver_mngr.save(sess, global_step=global_step)
                            print('saving model at: %s'%saver_mngr._name) 
                    if len(validation_loss)  > 5:
                        if (np.min(validation_loss) not in validation_loss[-5:]):
                            tf_train_utils.change_lr_val(sess, learning_rate_var, 0.9) 
                            saver_mngr.restore(sess)
                            print('decreasing learning_rate to\
                                    {}'.format(learning_rate_var.eval()))
                    print('valid loss: %f recon loss: %f encoded sparsity: %f' %
                        (valid_loss, recon, valid_sparsity_out))
                    summary = sess.run(merged, feed_dict)
                    valid_summ_writer.add_summary(summary, global_step=global_step.eval(session=sess))

            print('epoch %d: loss val:%f' % (epocha, HYPR_PARAMS['batch_size']  * epoch_loss / X_train.shape[0]))

        #run test
        save_path = DIR_PATH + '/logdir/'
        test_loss = evaluate.test(sess, _model, test_dh, loss, save_path)
        decoder_filters = np.squeeze(_model.decoder.convdict.eval(session=sess))
    
    print('='*40 + '\nTEST LOSS {}\n'.format(test_loss) + '='*40)
    plt.figure()
    plot_num = np.sqrt(HYPR_PARAMS['kernel_count'])
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
    global HYPR_PARAMS
    HYPR_PARAMS = vars(lista_args.args(train_mode=True))
    main()
