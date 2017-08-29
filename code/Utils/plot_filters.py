import matplotlib.pyplot as plt 
import numpy as np


def plotfromfile(filter_path_npy, key=None):
    flts = np.load(filter_path_npy)
    if key is not None:
        flts = flts[key]
    plotfromdata(flts)

def plotfromdata(data):
    flts = np.squeeze(data)
    plt.figure()
    if len(flts.shape) == 2: # image not filters
        plt.imshow(flts, cmap='gray')
    else:
        flts = np.transpose(flts, [2, 0, 1])
        dim = np.sqrt(flts.shape[0])
        plt.ion()
        for id, f in enumerate(flts):
            plt.subplot(dim, dim, id + 1)
            plt.imshow(f, cmap='gray', interpolation='bilinear')
    plt.show()


