import matplotlib.pyplot as plt 
import numpy as np


def plotfromfile(filter_path_npy, key=None, cmap='gray'):
    flts = np.load(filter_path_npy)
    if key is not None:
        flts = flts[key]
    plotfromdata(flts, cmap)

def plotfromdata(data, cmap):
    flts = np.squeeze(data)
    print(flts.shape)
    plt.figure()
    if len(flts.shape) == 2 and cmap=='gray' or len(flts.shape) == 3 and \
    not cmap =='gray' : # image not filters
        plt.imshow(flts, cmap=cmap)
    else:
        flts = np.transpose(flts, [2, 0, 1])
        dim = np.sqrt(flts.shape[0])
        plt.ion()
        for id, f in enumerate(flts):
            plt.subplot(dim, dim, id + 1)
            plt.imshow(f,cmap=cmap, interpolation='bilinear')
            plt.xticks(range(0), [], color='white')
            plt.yticks(range(0), [], color='white')
    plt.show()


