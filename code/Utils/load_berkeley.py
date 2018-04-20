import tarfile as tar
import numpy as np
import random
from PIL import Image
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BSDSPATH=os.path.join(FILE_PATH, '..', '..', 'images', 'BSDS300-images.tgz') 
NP_BSDSPATH_GRAY=os.path.join(FILE_PATH, '..', '..', 'images', 'BSDS300-images.npz') 

def build_dataset(pathtosave, rgb2gray=True, ims_path=BSDSPATH):

    X_train = []
    X_test = []
    X_val = []
    dx = dy = 321

    db_p = tar.open(ims_path)
    for f_info in db_p.getmembers():
        print(f_info.name)
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

    if not os.path.isdir(os.path.dirname(pathtosave)):
        os.makedirs(os.path.dirname(pathtosave))
    np.savez(pathtosave, TRAIN=X_train, TEST=X_test, VAL=X_val)

def load(grayscale=True, path=NP_BSDSPATH_GRAY):
    if not os.path.isfile(path):
        build_dataset(path, rgb2gray=grayscale)        
    if grayscale:
        DB = np.load(path)
    else:
        raise NotImplementedError('creat an npz dict of RGB imgs')

    print('trainset size: %d, validset size: %d'%(len(DB['TRAIN']), len(DB['TEST'])))
    return DB['TRAIN'], DB['TEST']

