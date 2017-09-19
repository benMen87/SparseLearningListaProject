import tarfile as tar
import numpy as np
import random
from PIL import Image
import os
import sys

BSDSPATH='/home/hillel/projects/SparseLearningListaProject/images/BSR_bsds500.tgz' 
NP_BSDSPATH_GRAY='/home/hillel/projects/SparseLearningListaProject/images/ber_nex_321X321.npz' 

def load(grayscale):
    if grayscale:
        DB = np.load(NP_BSDSPATH_GRAY)
    else:
        raise NotImplementedError('creat an npz dict of RGB imgs')

    # use val set a test set...
    return np.concatenate((DB['TRAIN'], DB['TEST']), axis=0), DB['VAL']


def build_dataset(pathtosave, rgb2gray):

    X_train = []
    X_test = []
    X_val = []
    dx = dy = 321

    db_p = tar.open(BSDSPATH)
    for f_info in db_p.getmembers():
        if f_info.name.endswith('.jpg'):
            img_fp = db_p.extractfile(f_info)
            I = Image.open(img_fp)
            w, h = I.size
            if rgb2gray:
                x = y = 0
                if not w == dx:
                    x = random.randint(0, w-dx-1)
                if not h == dy:
                    y = random.randint(0, h-dy-1)
                I = I.convert('L')
                I = np.asarray(I.crop((x, y, x+dx, y+dy)))
                I = I[...,np.newaxis]
            else:
                I = np.asarray(I)
            if I.shape[0] > I.shape[1]:
                I = np.transpose(I, [1, 0, 2])
            if 'train' in f_info.name:
                X_train.append(I)
            elif 'val' in f_info.name:
                X_val.append(I)
            else: # test
                X_test.append(I)

    if not os.path.isdir(pathtosave):
        os.makedirs(pathtosave)
    np.savez(pathtosave + 'berkeley_db_321X321', TRAIN=X_train, TEST=X_test, VAL=X_val)

