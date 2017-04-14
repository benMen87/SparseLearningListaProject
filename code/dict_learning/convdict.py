import os
import numpy as np
from scipy.linalg import toeplitz

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def build_convdict(signal_size, filter_size=9, filter_count=16):

    if filter_size % 2 == 0:
        raise NotImplemented('Only odd filters are implented')

    filter_array = [np.random.randn(filter_size) for _ in range(filter_count)]
    filter_array = [f/np.linalg.norm(f) for f in filter_array]

    D = []
    for f in filter_array:
        row = np.append(f[(filter_size - 1) // 2:], np.zeros(signal_size - (filter_size + 1) / 2))
        col = np.append(f[:(1 + (filter_size - 1) // 2)][::-1], np.zeros(signal_size - (filter_size + 1) / 2))
        D.append(toeplitz(col, row).T)
    D = np.concatenate(D, axis=1)

    return D, filter_array


def test_build_convdict():
    filter_size = 3
    filter_count = 2
    signal_size = 5

    x = np.random.randn(signal_size)
    D, filter_array = build_convdict(signal_size=signal_size,
                                     filter_size=filter_size,
                                     filter_count=filter_count)

    y = np.convolve(x, filter_array[0][::-1], 'SAME')
    yy = np.matmul(D, x)

    assert np.allclose(y ,yy) == True, 'ERROR!'

    print('PASSED')

if __name__ == '__main__':
    filter_size = 9
    filter_count = 6
    signal_size = 100

    D, filter_array = build_convdict(signal_size=signal_size,
                                     filter_size=filter_size,
                                     filter_count=filter_count)
    np.save(DIR_PATH+'/../../covdict_data/conv_dict', D)
    np.save(DIR_PATH+'/../../covdict_data/filter_arr', filter_array)