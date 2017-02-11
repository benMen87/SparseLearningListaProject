import numpy as np
import tarfile as tar
from PIL import Image
import matplotlib.pyplot as plt
import os
import cod
import scipy.io


def run_cod_on_patches(I, patch_size, dict_size):
    n, m = I.shape
    #TODO: smart patch extractor
    patch = I[int(n/2):int(n/2) + patch_size,
            int(m/2):int(m/2) + patch_size]
    patch = patch.reshape(patch_size ** 2, 1)
    p_mean = np.mean(patch)
    patch = patch - p_mean
    Wd_init = np.random.randn(patch_size ** 2, dict_size)
    col_norm = np.linalg.norm(Wd_init, axis=0)
    Wd_init = Wd_init / col_norm

    #
    #for testing with matlab
    scipy.io.savemat('m_ref/Wd.mat', mdict={'Wd':Wd_init})
    scipy.io.savemat('m_ref/X.mat', mdict={'X':patch})


    sparse_cod = cod.CoD(Wd_init)
    sparse_cod.run_cod(patch)



def train_cod(image_dirpath):
    with tar.open(image_dirpath) as img_tar:
        img = img_tar.extractfile(img_tar.getmembers()[3])
        I = Image.open(img)
        I = np.asarray(I.convert('L'))

        run_cod_on_patches(I, 10, 100)
        

if __name__ == '__main__':
    train_cod('images/BSDS300-images.tgz')

   # I = Image.open('./images/BSDS300/images/test/101085.jpg')
   # I = np.asarray(I.convert('L'))
   # n, m = I.shape
   # patch_size = 10
   # patch = I[int(n/2):int(n/2) + patch_size,
        #int(m/2):int(m/2) + patch_size]
   #patch = patch.reshape(patch_size*patch_size, 1
   #p_mean = np.mean(patch)
   #patch = patch - p_mean
   #Wd_init = np.random.rand(patch_size ** 2, dict_size)
   #col_norm = np.linalg.norm(Wd_init, axis=0)
   #Wd_init = Wd_init / col_norm

