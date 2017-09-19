import os
import numpy as np
import glob
from PIL import Image
from resizeimage import resizeimage

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def rgb2gray(X):
    if X.shape[-1] == 1:
        return X
    r, g, b = X[...,0], X[...,1], X[...,2]
    return (0.2125 * r) + (0.7154 * g) + (0.0721 * b)

def load_fruit(grayscale=False):
    FRUIT_PATH = FILE_PATH +  '/../../../FastFlexibleCSC_Code_Heide2015/datasets/Images/fruit_100_100/*'
    return load(FRUIT_PATH, grayscale)    

def load(searchpath, grayscale=False):
    
    X = []
    for im_path in glob.glob(searchpath):
        X.append(np.array(Image.open(im_path)))
        # X.append(np.array(resizeimage.resize_cover(Image.open(im_path), [96, 96, -1])))
    X = np.array(X)

    if len(X.shape) == 3 and X.shape[-1] > 1:
        X = X[...,np.newaxis]

    if grayscale:
        X = rgb2gray(X)
        if len(X.shape) == 2:
            X = X[..., np.newaxis]
    return np.array(X)

#load_fruit()
load('/home/hillel/projects/FastFlexibleCSC_Code_Heide2015/datasets/Images/singles/supplement/reconstruction/results_rec_dataset_fruit/clean*.png')

