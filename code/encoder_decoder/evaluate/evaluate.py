import sys
import os
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt
import time

DIR_PATH = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(os.path.abspath(DIR_PATH))
sys.path.append(os.path.abspath(DIR_PATH) + '/../')

from Utils import psnr
from approx_sae.approx_conv2d_sparse_ae import ApproxCSC

DEFAULT_IMAGE_PATH = '/data/hillel/data_sets/test_images/'
MODEL_DIR = \
    '/home/hillel/projects/dev/SparseLearningListaProject/code/encoder_decoder/logdir/models/u10_kc64_ks7_sf0.1/'

def infer(_sess, feed_dict, eval_list):
    _eval_list = _sess.run(eval_list, feed_dict)
    return _eval_list

def test(_sess, _model, batch_gen, loss, savepath):
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
        axarr[1,1].set_title('noisy image -  psnr: {0:.3} [db]'.format(psnr.psnr(_o, _n)))
        axarr[1,1].axis('off')

        axarr[1,0].imshow(np.squeeze(_r), cmap=cmap)
        axarr[1,0].set_title('reconstructed image -  psnr: {0:.3}\
                [db]'.format(psnr.psnr(_o, _r)))
        axarr[1,0].axis('off')
        example_ims = savepath + '/plots/' + str(_nm) + '.png'
        f.savefig(example_ims)
    plt.close()

class Args():
    def __init__(self):
        pass

def set_args():
    args = Args()
    args.load_pretrained_dict = False
    args.unroll_count = 10
    args.batch_size = 1
    args.kernel_size = 7
    args.kernel_count = 64
    args.shrinkge_type = 'soft thresh'
    args.inpaint = True
    args.dont_train_dict = False
    return args

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
   PSNR = 10 * np.log10(1/ MSE)
   return PSNR

def main(args, **kwargs):
    
    pad_size = args.kernel_size // 2
    tetspath = kwargs.get('testpath', DEFAULT_IMAGE_PATH) 
    testext = kwargs.get('ext', 'png')
    model_dir = kwargs.get('model_dir', MODEL_DIR)
    test_imgs = [load(img_path) for img_path in glob.glob(tetspath + '/*' + testext)]

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model  = ApproxCSC()
    model.build_model(
        unroll_count=args.unroll_count,
        L=1, batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        shrinkge_type=args.shrinkge_type,
        kernel_count=args.kernel_count,
        channel_size=1  # TODO: fixthis
    )
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    if kwargs.get('denoise', False):
        eps = kwargs.get('noise_factor', 20.0)
        test_imgs_n = [_n + np.random.normal(size=_n.shape, scale=(eps/255)) for _n in test_imgs ]

    test_imgs_n = [pad_image(_n, pad_size) for _n in test_imgs_n ]
    test_imgs_n = [_n[...,None] for _n in test_imgs_n]
    im_results, sc_results = infrence(_sess=sess, _model=model, _images=test_imgs_n)

    im_results = [np.squeeze(res[pad_size:-pad_size, pad_size:-pad_size,...]) for res in im_results]
    test_imgs_n = [np.squeeze(_n[pad_size:-pad_size, pad_size:-pad_size]) for _n in test_imgs_n]
    sc_results = [np.squeeze(_sc) for _sc in sc_results]

    save_figs(
        savepath=DIR_PATH + '/logdir/',
        _noisy=test_imgs_n,
        _result=im_results,
        _orig=test_imgs,
        _sc=sc_results)


if __name__ == '__main__':
    """
    From main run test on default images
    """
    args = set_args() 
    main(args, denoise=True)
