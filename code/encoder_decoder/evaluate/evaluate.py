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
import argparse

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(os.path.abspath(DIR_PATH))
sys.path.append(os.path.abspath(DIR_PATH) + '/../')

from Utils import psnr
from approx_sae.approx_conv2d_sparse_ae import ApproxCSC

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
    print('batch size {}'.format(batch_gen.batch_size))

    for X_batch, Y_batch in batch_gen:
        feed_dict = {
            _model.input: X_batch,
            _model.target: Y_batch
           # _model.encoder.mask: batch_gen.mask
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


def infrence(_sess, _model, _images, _mask=None):

    feed_dict = {}
    _eval_list = [_model.sparsecode, _model.output]
    im_results = []
    sc_results = []

    if not isinstance(_images, list):
        _images = [_images]

    for im in _images:
        feed_dict = {_model.input: [im]}
        if _mask is not None:
            feed_dict[encd_mask] =  _mask
        Z, im_hat = infer(_sess, feed_dict, _eval_list)
        np.clip(im_hat, 0, 1)  # clip values
        im_results.append(im_hat[0])
        sc_results.append(Z)
    return im_results, sc_results


def save_figs(_noisy, _result, _orig=[], _sc=[], savepath='./', _names=''):

    _save_np = savepath + '/data/example_ims'
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
        axarr[1,1].set_title('noisy image -  psnr: {0:.3}\
                [db]'.format(psnr.psnr(_o, _n, verbose=False)))
        axarr[1,1].axis('off')

        axarr[1,0].imshow(np.squeeze(_r), cmap=cmap)
        axarr[1,0].set_title('reconstructed image -  psnr: {0:.3}\
                [db]'.format(psnr.psnr(_o, _r)))
        axarr[1,0].axis('off')
        example_ims = savepath + '/plots/' + str(_nm) + '.png'
        f.savefig(example_ims)
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
    tetspath = args.test_path
    testext = 'png' # TODO: make as user input
    model_dir = args.model_path
    test_imgs = [load(img_path) for img_path in glob.glob(tetspath + '/*' + testext)]

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model  = ApproxCSC(type=args.model_type)
    model.build_model(
        amount_stacked=args.amount_stacked,
        unroll_count=args.unroll_count,
        L=1, batch_size=1,
        kernel_size=args.kernel_size,
        kernel_count=args.kernel_count,
        channel_size=1, #TODO: fixthis
        norm_kernal=args.norm_kernal
    )
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    if args.task == 'denoise':
        eps = args.noise_sigma
        test_imgs_n = [_n + np.random.normal(size=_n.shape, scale=(eps/255)) for _n in test_imgs ]

    test_imgs_n = [pad_image(_n, pad_size) for _n in test_imgs_n ]
    test_imgs_n = [_n[..., None] for _n in test_imgs_n]
    im_results, sc_results = infrence(_sess=sess, _model=model, _images=test_imgs_n)

    im_results = [np.squeeze(res[pad_size:-pad_size, pad_size:-pad_size,...]) for res in im_results]
    test_imgs_n = [np.squeeze(_n[pad_size:-pad_size, pad_size:-pad_size]) for _n in test_imgs_n]
    sc_results = [np.squeeze(_sc) for _sc in sc_results]

    psnr_avg = np.mean([eval_result(_o, _r) for _o, _r in zip(test_imgs,
        im_results)])

    save_figs(
        savepath=DIR_PATH + '/logdir/',
        _noisy=test_imgs_n,
        _result=im_results,
        _orig=test_imgs,
        _sc=sc_results)
    print('PSNR avrage is {}'.format(psnr_avg))

if __name__ == '__main__':
    """
    From main run test on default images
    """
    parser = argparse.ArgumentParser(description='evaluate model')
    parser.add_argument('-ks', '--kernel_size', default=7, type=int,
                                help='kernel size to be used in lista_conv')
    parser.add_argument('-kc', '--kernel_count', default=64, type=int,
                                help='amount of kernel to use in lista_conv')
    parser.add_argument('-u', '--unroll_count', default=3,
         type=int, help='Amount of Reccurent timesteps for decoder')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_DIR, type=str,
        help='path of cpk file')
    parser.add_argument('--task', default='multi_denoise', choices=['denoise',
        'inpaint'], help='add noise to input')
    parser.add_argument('--noise_sigma', '-ns', type=float, default=20,
            help='noise magnitude')
    parser.add_argument('--model_type', '-mt', default='untied', choices=['convdict', 'convmultidict', 'untied'])
    parser.add_argument('--norm_kernal', '-nk', action='store_true',
        help='keep kernels with unit norm') 
    parser.add_argument('--test_path', '-tp',
            default='/data/hillel/data_sets/test_images/')
    parser.add_argument('--amount_stacked', '-as', default=1, type=int)
    args = parser.parse_args()
    main(args)
