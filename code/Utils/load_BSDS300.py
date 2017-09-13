import tarfile as tar
import numpy as np
from PIL import Image
import os
import sys

BSDS300PATH='/home/hillel/projects/SparseLearningListaProject/images/BSR_bsds500.tgz' 

def load(rgb2gray=True):
    X_train = []
    X_test = []
    db_p = tar.open(BSDS300PATH)
    for f_info in db_p.getmembers():
        if f_info.name.endswith('.jpg'):
            img_fp = db_p.extractfile(f_info)
            I = Image.open(img_fp)
            if rgb2gray:
                I = np.asarray(I.convert('L'))
                I = I[...,np.newaxis]
            if I.shape[0] > I.shape[1]:
                I = np.transpose(I, [1, 0, 2])
            if 'train' in f_info.name:
                X_train.append(I)
            else: # test
                X_test.append(I)
    print np.array(X_train).shape
    return  np.array(X_train), np.array(X_test)

