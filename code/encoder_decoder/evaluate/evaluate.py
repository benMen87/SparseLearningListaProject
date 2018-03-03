import sys
import os
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(os.path.abspath(DIR_PATH))
sys.path.append(os.path.abspath(DIR_PATH) + '/../')

from Utils import psnr
from Utils import lista_args
from approx_sae.approx_conv2d_sparse_ae import ApproxCSC

import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'

DEFAULT_IMAGE_PATH = '/data/hillel/data_sets/test_images/'
DEFAULT_MODEL_DIR = \
    '/home/hillel/projects/dev/SparseLearningListaProject/code/encoder_decoder/logdir/models/u10_kc64_ks3_untied'

def infer(_sess, feed_dict, eval_list):
    _eval_list = _sess.run(eval_list, feed_dict)
    return _eval_list

def test(_sess, _model, batch_gen, loss, savepath):
    """
    Used on testset after training 
    """
    test_loss = 0
    test_iter = 0
    print('batch size {}'.format(batch_gen._batch_size))

    for X_batch, Y_batch in batch_gen:
        feed_dict = {
            _model.input: X_batch,
            _model.target: Y_batch,
#            _model.encoder.mask: (X_batch == Y_batch).astype(float)
        }
        test_loss += infer(_sess, feed_dict, loss)
        test_iter += 1

    test_loss = test_loss/test_iter

    # for debug save We/Wd
    model_params = savepath + 'data/modelParams'
    _We, _Wd, _theta, _convdict = _sess.run([
        _model.encoder.We, 
        _model.encoder.Wd,
        _model.encoder.theta,
        _model.decoder.convdict
        ])
    np.savez(model_params, Wd=_Wd, We=_We, theta=_theta, convdict=_convdict)

    #last batch save image results of last batch
    im_res, sc = infer(_sess, feed_dict, [_model.output, _model.sparsecode])
    save_figs(X_batch, im_res, Y_batch, _sc=sc, savepath=savepath)

    return test_loss


def infrence(_sess, _model, _images, _mask=None, task='denoise', task_spesific_args_list=[]):

    feed_dict = {}
    _eval_list = [_model.sparsecode, _model.output]
    im_results = []
    sc_results = []
    img_n      = []

    if not isinstance(_images, list):
        _images = [_images]

    if 'deblur' in task:
        _eval_list.append(_model.encoder._inputs_blur)

    if 'dynamic' in task and 'denoise' in task:
        feed_dict[_model.encoder.sigma] = task_spesific_args_list[0]


    for im in _images:
        feed_dict[_model.input] =  [im]
        if _mask is not None:
            feed_dict[encd_mask] =  _mask
        if 'denoise' in task:
            Z, im_hat = infer(_sess, feed_dict, _eval_list)
        elif 'deblur' in task:
            Z, im_hat, blur_im = infer(_sess, feed_dict, _eval_list)
            img_n.append(blur_im[0,...])

        np.clip(im_hat, 0, 1)  # clip values
        im_results.append(im_hat[0])
        sc_results.append(Z)

    return im_results, sc_results, img_n

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_figs(_noisy, _result, _orig=[], _sc=[], savepath='./', _names=''):

    save_data_path = savepath + '/data/'
    save_plots_path = savepath + '/plots'
    maybe_create_dir(save_data_path)
    maybe_create_dir(save_plots_path)

    _example_im_fp = lambda id: savepath + '/plots/' + str(id) + '.png'
    _save_np = save_data_path  + '/example_ims'
    np.savez(_save_np, X=_noisy, Y=_orig, Z=_sc, IM_hat=_result)
    print('saved example img data de/en at %s' % _save_np)

    if _orig is None:
        _orig = [None] * len(_result)

    if _names == '':
        _names = range(0, len(_noisy))

    f, axarr = plt.subplots(2, 2)
    cmap = 'gray' #if args.grayscale else 'viridis'
    for _o, _n, _r, _nm in zip(_orig, _noisy, _result, _names):
        _o = _o if _o is not None else np.zeros_like(_n)
        axarr[0,1].axis('off')
        axarr[0,0].imshow(np.squeeze(_o), cmap=cmap)
        axarr[0,0].set_title('original image')
        axarr[0,0].axis('off')

        axarr[1,1].imshow(np.squeeze(_n), cmap=cmap)
        axarr[1,1].set_title('noisy image -  psnr: {0:.3}'.format(psnr.psnr(_o, _n, verbose=False)))
        axarr[1,1].axis('off')

        axarr[1,0].imshow(np.squeeze(_r), cmap=cmap)
        axarr[1,0].set_title('reconstructed image -  psnr: {0:.3}'.format(psnr.psnr(_o, _r)))
        axarr[1,0].axis('off')
        f.savefig(_example_im_fp(_nm))
    plt.close()

def load(img_path):
    I_orig = Image.open(img_path)
    I = np.asarray(I_orig)
    if len(I.shape) == 3:
        I = rgb2gray(I)
    I = I.astype('float32') / 255
    return I

def pad_image(I, pad_size):
    I_p = np.pad(I, pad_size, 'reflect')
    return I_p

def eval_result(I_orig, I_hat):
   MSE = np.mean((I_orig - I_hat)**2)
   N = np.prod(I_orig.shape)
   PSNR = 10 * np.log10(np.max(I_orig)/ MSE)
   return PSNR

def main(args):
    """
    Use to evaluate trained model on famous images (Lena etc.)
    """
    pad_size = args.kernel_size // 2
    tetspath = args.image_path
    extention = args.patt
    model_dir = args.checkpoint_path
    print(model_dir)
    test_imgs = [load(img_path) for img_path in glob.glob(tetspath + '/*' + extention)]

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # release constraint of training  HW 

    model  = ApproxCSC(type=args.model_type)
    model.build_model(
        amount_stacked=args.amount_stacked,
        unroll_count=args.unroll_count,
        L=1, batch_size=1,
        kernel_size=args.kernel_size,
        kernel_count=args.kernel_count,
        channel_size=1, #TODO: fixthis
        norm_kernal=args.norm_kernal,
        is_train=False,
        psf_id=args.psf_id
    )
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    if 'denoise' in args.task:
        eps = args.noise_sigma
        test_imgs_n = [_n + np.random.normal(size=_n.shape, scale=(eps/255)) for _n in test_imgs ]
    elif 'deblur' in args.task:
        test_imgs_n = test_imgs # images are corupted as part of input model

    test_imgs_n = [pad_image(_n, pad_size) for _n in test_imgs_n ]
    test_imgs_n = [_n[..., None] for _n in test_imgs_n]

    im_results, sc_results, test_imgs_n = infrence(_sess=sess, _model=model, _images=test_imgs_n, task=args.task)#, task_spesific_args_list=[eps])
    im_results = [np.squeeze(res[pad_size:-pad_size, pad_size:-pad_size,...]) for res in im_results]
    test_imgs_n = [np.squeeze(_n[pad_size:-pad_size, pad_size:-pad_size]) for _n in test_imgs_n]
    sc_results = [np.squeeze(_sc) for _sc in sc_results]

    psnr_avg = np.mean([eval_result(_o, _r) for _o, _r in zip(test_imgs,
        im_results)])
    print('PSNR avrage is {}'.format(psnr_avg))

    savepath = args.savepath if args.savepath is not ' ' else DIR_PATH + '/logdir/'
    save_figs(
        savepath=savepath,
        _noisy=test_imgs_n,
        _result=im_results,
        _orig=test_imgs,
        _sc=sc_results)
    
if __name__ == '__main__':
    """
    From main run test on default images
    """
    main(lista_args.args())
