import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def display_atoms(Wd, patch_size):

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(Wd[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.ion()
    plt.show()


def buildDCT(b_dim, K):
    Pn = np.uint32(np.ceil(np.sqrt(K)))
    DCT = np.zeros((b_dim, Pn))

    for k in range(Pn):
        V = np.cos(np.arange(0, b_dim).T * k * np.pi / Pn)
        if k > 0:
            V = V - np.mean(V)
        DCT[:, k] = V / np.linalg.norm(V)
    return np.kron(DCT, DCT)

atom_count = 600
patch_size = 10
DCT = buildDCT(patch_size, atom_count)
DCT = DCT[:,0:atom_count]
np.save(DIR_PATH + '/../../dct_data/Wd', DCT)
