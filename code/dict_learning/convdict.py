import os
import numpy as np
from scipy.linalg import toeplitz
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def build_convdict(signal_size, filter_array=None, filter_size=9, filter_count=16):

    if filter_size % 2 == 0:
        raise NotImplemented('Only odd filters are implented')

    if filter_array is None:
        filter_array = [np.random.randn(filter_size)  for _ in range(filter_count)]
        filter_array = [f/np.linalg.norm(f) for f in filter_array]

    D = []
    for f in filter_array:
        row = np.append(f[(filter_size - 1) // 2:], np.zeros(signal_size - (filter_size + 1) / 2))
        col = np.append(f[:(1 + (filter_size - 1) // 2)][::-1], np.zeros(signal_size - (filter_size + 1) / 2))
        D.append(toeplitz(col, row))
    D = np.concatenate(D, axis=1)

    return D, filter_array


def matcovd2d(kernal_2d, sig_rows, sig_cols):

    k_n, k_m = kernal_2d.shape
    sig_stack_len = sig_rows*sig_cols
    blocks = [build_convdict(sig_cols, [f], len(f), 1)[0] for f in kernal_2d]
    T = np.zeros((sig_rows,
                  blocks[0].shape[0],
                  sig_rows,
                  blocks[1].shape[1]))

    for i in range(len(blocks)):
        for j in range(sig_rows):
            b = (i + (k_m-1)/2) % len(blocks)
            num_diag = (k_m-1)/2 - b
            if j + np.abs(num_diag) < sig_rows:
                if num_diag < 0:
                    T[j, :, j - num_diag, :] = blocks[b]
                else:
                    T[j + num_diag, :, j, :] = blocks[b]

    # SAME convolution
    # T = T[]
    T.shape = (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3])
    return T


def build_convdict2d(filter_array=None, filter_count=16, filter_shape=(3,3), sig_shape=(10,10)):

    if filter_array is None:
        filter_array = [np.random.randn(filter_shape[0], filter_shape[1]) for _ in range(filter_count)]
        filter_array = [f/np.linalg.norm(f) for f in filter_array]
        
        D = []
        for f in filter_array:
            D.append(matcovd2d(f, sig_shape[0], sig_shape[1]))
        D = np.concatenate(D, axis=1)
        return D, filter_array

if __name__ == '__main__':

    sig_shape = (10, 10)
    filter_shape = (3, 3)
    filter_count = 6
    D, filter_array = build_convdict2d(sig_shape=sig_shape,
                                     filter_shape=filter_shape,
                                     filter_count=filter_count)
    np.save(DIR_PATH+'/../../convdict2d_data/Wd', D)
    np.save(DIR_PATH+'/../../convdict2d_data/filter_arr', filter_array)