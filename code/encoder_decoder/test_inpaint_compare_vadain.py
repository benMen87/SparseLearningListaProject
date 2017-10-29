import scipy.io as scio
import numpy as np
import tensorflow as tf 
import test_inpaint
import sparse_aed





IMS = scio.loadmat('./local_cn_ims.mat')
IMS = [np.transpose(np.squeeze(IMS['b']), [3,0,1]),
        np.transpose(np.squeeze(IMS['b1']), [3,0,1])]

arg = test_inpaint.set_args()
pad_size = args.kernel_size // 2
sess = tf.Session()
encoder, decoder, mask = sparse_aed.build_model(args, [None, None, 1])
model = test_inpaint.Model(encoder=encoder, decoder=decoder, mask=mask)

for im_type in IMS:
    for img in im_type:
        I = img.copy()
        _min = np.min(I)
        _max = np.max(I)
        I = (I - _min) / (_max - _min)
        I_n, _mask = inpaint(I, pad_size)
        I_hat = infrence(sess, model, I_n, mask=_mask)
        I_hat = np.clip(I_hat, 0, 1)
        I_hat = I_hat[pad_size:-pad_size, pad_size:-pad_size]
        I_n = I_n[pad_size:-pad_size, pad_size:-pad_size,0]
         
