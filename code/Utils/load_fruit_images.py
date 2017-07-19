import os
import numpy as np
import glob
from PIL import Image
from resizeimage import resizeimage

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
FRUIT_PATH = FILE_PATH +  '/../../../FastFlexibleCSC_Code_Heide2015/datasets/Images/fruit_100_100/'

print(FRUIT_PATH)

def load():
    X = []
    for im_path in glob.glob(FRUIT_PATH+'*.jpg'):
	X.append(np.array(resizeimage.resize_cover(Image.open(im_path), [96, 96])))
    return np.array(X)

