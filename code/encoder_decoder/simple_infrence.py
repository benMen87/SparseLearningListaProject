import sys
import tensorflow as tf
import numpy as np
import sparse_aed
from PIL import Image
import matplotlib.pyplot as plt

MODEL_DIR = '/home/hillel/projects/lista_multi_res/SparseLearningListaProject/code/encoder_decoder/logdir/models/denoiseMultiResU15_HiRes_pascal_5_7_11_norm_encoder'
IMAGE = '/home/hillel/projects/SparseLearningListaProject/code/encoder_decoder/test_images/images/Lena512.png'

class Args():
    def __init__(self):
        pass

def set_args():
    args = Args()
    args.load_pretrained_dict = False
    args.unroll_count = 15
    args.batch_size = 1
    args.kernel_size = 5
    args.kernel_count = 36
    args.shrinkge_type = 'soft thresh'
    args.inpaint = False
    args.dont_train_dict = False
    return args

def load_and_shift(shifts):
    input_shape = [512, 512, 1]
    
    I_orig = Image.open(IMAGE)
    I = np.asarray(I_orig.convert('L'))
    I = I.astype('float32') / 255
    I_n = I[..., np.newaxis] + np.random.normal(0, 20.0 / 255, input_shape)

    I_in = np.zeros([2*len(shifts)] + input_shape)

    for id, sh in enumerate(shifts):
        I_in[2*id, sh[0]:, sh[1]:] =\
                I_n[:input_shape[0] - sh[0], :input_shape[1] - sh[1],:]
        I_in[2*id + 1, :input_shape[0] - sh[0], :input_shape[1] - sh[1]] =\
                I_n[sh[0]:, sh[1]:, :]

    return I, I_in

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

def denoise():
    
    shifts = [(0,0), (3,3), (3, 0), (0, 3)]

    I_clean, I_in = load_and_shift(shifts)
    args = set_args()
    args.batch_size = I_in.shape[0]
    
    sess = tf.Session()
    encoder, decoder, _ = sparse_aed.build_model(args, [512, 512, 1])
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))


    feed_dict = {encoder.input: I_in}
    I_batch = np.squeeze(sess.run(decoder.output, feed_dict))
    I_hat = unshift(I_batch, shifts)
    Image.fromarray(np.uint8(I_hat * 255)).save('shift_res.png')
    plt.imshow(I_hat, cmap='gray'); plt.show()
    psnr = 10 * np.log10(1 / np.mean((I_clean - I_hat)**2) )
    print('PSNR {}'.format(psnr))
    np.savez('use_shifs', I=I_clean, I_in=I_in, I_batch=I_batch, I_hat=I_hat)

if __name__ == '__main__':
    print('Running denoise')
    denoise()
    print('Done')
