import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plotfromfile(filter_path_npy, key=None, cmap='gray'):
    flts = np.load(filter_path_npy)
    if key is not None:
        flts = flts[key]
    plotfromdata(flts, cmap)

def plotfromdata(data, cmap):
    flts = np.squeeze(data)
    print(flts.shape)
    
    if len(flts.shape) == 2 and cmap=='gray' or len(flts.shape) == 3 and \
    not cmap =='gray' : # image not filters
        plt.figure()
        plt.imshow(flts, cmap=cmap)
    else:
        flts = np.transpose(flts, [2, 0, 1])
        dim = np.sqrt(flts.shape[0]).astype('int')
        plt.figure(figsize=(dim-1, dim-1))
        plt.ion()
        gs = gridspec.GridSpec(dim, dim, wspace=0.0, hspace=0.0, 
            top=1.-0.5/(dim+1), bottom=0.5/(dim+1), 
            left=0.5/(dim+1), right=1-0.5/(dim+1))

        for i in range(dim):
            for j in range(dim):
                f = flts[i*dim + j]
                ax = plt.subplot(gs[i,j])
                ax.imshow(f, cmap='gray', interpolation='bilinear')
                plt.xticks(range(0), [], color='white')
                plt.yticks(range(0), [], color='white')
    plt.show()

   # for id, f in enumerate(flts):
   #         plt.subplot(dim, dim, id + 1)
   #         plt.imshow(f,cmap=cmap, interpolation='bilinear')
   #         plt.xticks(range(0), [], color='white')
   #         plt.yticks(range(0), [], color='white')
   # plt.show()


