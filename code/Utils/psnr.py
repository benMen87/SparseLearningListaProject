import numpy as np
import sys



def psnr(im, recon, verbose=True):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    MSE = np.sum((im - recon)**2) / (im.shape[0] * im.shape[1])
    MAX = np.max(im)
    PSNR = 10 * np.log10(MAX ** 2 / MSE)
    if verbose:
        print('PSNR %f'%PSNR)
    return PSNR


