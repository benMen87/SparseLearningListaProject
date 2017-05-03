import numpy as np
import convdict
from scipy import signal

def test_build_convdict():
    filter_size = 3
    filter_count = 1
    signal_size = 4

    x = np.random.randn(signal_size)
    D, filter_array = convdict.build_convdict(signal_size=signal_size,
                                     filter_size=filter_size,
                                     filter_count=filter_count)

    y = np.convolve(x, filter_array[0][::-1], 'SAME')
    yy = np.matmul(D, x)

    assert np.allclose(y ,yy) == True, 'ERROR!'

    print('PASSED')


def test_2dcovdict():

    cov_dict_list = []
    filter_2dsize = 3
    signal_nrows = 10
    signal_ncols = 10
    sig = np.random.rand(signal_nrows, signal_ncols)
    ker = np.array([[1,2,3],[4,5,6], [7,8,9]])

    D = convdict.matcovd2d(ker[::-1, ::-1], signal_nrows, signal_ncols)

    res1 = signal.convolve2d(sig, ker, mode='same')
    res2 = np.matmul(D, sig.reshape(signal_ncols*signal_nrows, 1))

    assert np.allclose(res1, res2.reshape(10,10)) == True, 'ERROR!'


if __name__ == '__main__':
    filter_size = 9
    filter_count = 6
    signal_size = 100

    test_2dcovdict()
