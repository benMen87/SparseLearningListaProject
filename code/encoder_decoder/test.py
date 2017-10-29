import sys
import os
import tensorflow as tf
import glob
import numpy as np
import sparse_aed
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt
import time

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = \
    '/home/hillel/projects/SparseLearningListaProject/code/encoder_decoder/logdir/models/u3_ks7_kc64_L1SSIM_inpaint'
IMAGES_DIR = '/home/hillel/projects/supplement/reconstruction/test_set/'


def rgb2gray(X):
    if X.shape[-1] == 1:
        return X
    r, g, b = X[...,0], X[...,1], X[...,2]
    return (0.2125 * r) + (0.7154 * g) + (0.0721 * b)


class Args():
    def __init__(self):
        pass

class Model():
    def __init__(self, encoder, decoder, mask):
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask

def set_args():
    args = Args()
    args.load_pretrained_dict = False
    args.unroll_count = 3
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

def unshift(I_batch, shifts):
    I_out = np.zeros(I_batch.shape[1:])
    input_shape = I_batch.shape[1:]
    for id in range(I_batch.shape[0]//2):
        sh = shifts[id]

        I_out[:input_shape[0] - sh[0], :input_shape[1] - sh[1]] +=\
                I_batch[2*id, sh[0]:, sh[1]:]
        I_out[sh[0]:, sh[1]:] += I_batch[2*id + 1, :input_shape[0] - sh[0], :input_shape[1] - sh[1]]

    I_out /= (I_batch.shape[0])
    return I_out

def infrence(sess, model, I_n, mask):
    # I_n = np.flipud(I_n)
    flipudlr = lambda I: np.fliplr(np.flipud(I))
    idendtity = lambda I: I
    flip_fns = [idendtity]#, flipudlr, np.fliplr, np.flipud]
    I_hat = 0
    for flip_fn in flip_fns:
        _I_n = flip_fn(I_n)
        _mask = flip_fn(mask)
        feed_dict = {model.encoder.input: [_I_n]}
        if not (mask == 1).all():
            feed_dict[model.mask] = [_mask]
        _I_hat = np.squeeze(sess.run(model.decoder.output, feed_dict))
        I_hat += flip_fn(_I_hat)
    I_hat /= len(flip_fns)
    return I_hat

def test():

    res_path = FILE_PATH + '/results/'
    if not os.path.exists(res_path):
         os.makedirs(res_path)

    args = set_args()
    images_paths = glob.glob(IMAGES_DIR + '*.png')
    print(images_paths)
    pad_size = args.kernel_size // 2
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    encoder, decoder, mask = sparse_aed.build_model(args, [None, None, 1])
    model = Model(encoder=encoder, decoder=decoder, mask=mask)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    for img_path in images_paths:
        im_name = os.path.splitext(os.path.basename(img_path))[0]
        I_orig = load(img_path)
        I = I_orig.copy()

        if not args.inpaint:
            I_n = I + np.random.normal(size=I.shape, scale=(20.0/255))
            _mask = 1
        else:
            keep_prob = 0.5
            _mask = np.random.choice([0, 1], size=I.shape, p=[1 - keep_prob,
                    keep_prob])
            I_n = I * _mask
            _mask = pad_image(_mask, pad_size)
            _mask = _mask[...,np.newaxis]

        I_n = pad_image(I_n, pad_size)
        I_n = I_n[...,np.newaxis]
        start_time = time.time()
        I_hat = infrence(sess, model, I_n, mask=_mask)
        print("----- %s --- %s seconds ---" % (im_name, (time.time() -start_time)))
        I_hat = np.clip(I_hat, 0, 1)
        I_hat = I_hat[pad_size:-pad_size, pad_size:-pad_size]
        I_n = I_n[pad_size:-pad_size, pad_size:-pad_size,0]
        psnr = eval_result(I_orig, I_hat)
        Image.fromarray(np.uint8(I_hat*255)).save( res_path + im_name + '_res'+str(psnr) + '.png')
        Image.fromarray(np.uint8(I_n*255)).save( res_path + im_name +
                '_n.png')
        scio.savemat(res_path + im_name, {'I': I*255,
                'I_n': I_n*255, 'I_hat': I_hat*255})  

       # print('IMGE: {} PSNR {}'.format(im_name, psnr))

if __name__ == '__main__':
    print('Running denoise')
    test()
    print('Done')
