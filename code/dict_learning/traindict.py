import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import Lasso
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(dir_path + '/..')
from sparse_coding.cod import CoD

class Traindict():

    """Learn dictionary that minizies lasso loss
       loss = ||X - WdZ||_2^2 + alpha||Z||_1 
       usind coordinate descent (i.e. CoD)
       note: this has been written to specifcly implement paper: 
       Learning Fast Approximations of Sparse Coding
    """

    def __init__(self, patch_size=10, Wd_init=None, atom_count=100, 
                 alpha=0.1, step_size=1.0, max_iter=2000):
        
        self.atom_count  = atom_count
        self.alpha       = alpha
        self.patch_size  = patch_size
        self.eta         = step_size
        self.max_iter    = max_iter

        if Wd_init is None:
            self.Wd_init = np.random.randn(patch_size ** 2, atom_count)
            col_norm     = np.linalg.norm(self.Wd_init, axis=0)
            self.Wd_init = self.Wd_init / col_norm
        else:
            self.Wd_init = Wd_init
               
    def learn_Wd_gd(self, train_data):
        """
        Learn sparse dictionary using gradient decent
        INPUT: train_data - feture_vector X sample i.e. [data_len, train_size]
        """
        Wd = self.Wd_init
        eta = lambda t: self.eta / (t + 1)

        loss = lambda Wd,X,Z: 0.5*np.linalg.norm(X - np.matmul(Wd, Z))**2 + self.alpha*np.linalg.norm(Z, 1) 
        grad = lambda Wd, X, Z: np.outer((np.matmul(Wd, Z) - X), Z)
        loss_arr = [np.inf]
        sparse_cod = CoD(Wd, alpha=self.alpha)

        #test with sklearn cod model
        #clf = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=1e-4)

        data_len, train_size = train_data.shape
        train_data_shuff = train_data[:, np.random.permutation(train_size)]
        for i in range(self.max_iter):

            patch = train_data_shuff[:, np.mod(i, train_size)]

            if patch.ndim == 1:
                patch = patch[:, np.newaxis]
                           
            Z, _ = sparse_cod.run_cod(patch, Wd)

            #Z = clf.fit(Wd, patch).coef_

            if Z.ndim == 1:
                Z = Z[:, np.newaxis]
            print('iter: %d'%i)

            #
            # optimize for given sparse code
            #t = 0
            #while True:
            loss_arr.append(loss(Wd, patch, Z))
            grad_t = grad(Wd, patch, Z)
            grad_norm = np.linalg.norm(grad_t, 'fro')
            dL = loss_arr[-2] - loss_arr[-1]

            #print('loss: %f'%loss_arr[-1])
            #print('grad: %f'%grad_norm)
            
            #
            #update dict
            Wd = Wd - eta(i+1)*grad_t
            Wd = Wd / np.linalg.norm(Wd, axis=0)

            if dL < 10e-6 * loss_arr[-1] and False:
                break 

        return (Wd, loss_arr)

def display_atoms(Wd, patch_size):

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(Wd[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
           interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.show()


from Utils import db_tools
import scipy.io as scio

if __name__ == '__main__':

    """
    1. Load train patches from DB or Create set from large image DB.
    2. Learn Sparse Dictionary using stochastic gradient decent and find sparse rep using - Li & Osher coordinate decent.
    3. Save Dictionary.
    """
    
    #
    # Hyper parameters
    PATCH_SIZE  = (10, 10)
    STD_THRESH  = 6 
    TRAIN_SIZE  = 3000 #amount of actual patches 
    MAX_ITER    = 2000
    ALPHA       = 0.5

    #
    # Load patches
    db_fp       = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\images\\BSDS300-images.tgz' 
    train_fp    = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\train_dict\\train_set.npy'.format(STD_THRESH) 
    train_data  = db_tools.load_maybe_build_train_set(train_fullpath=train_fp, db_fullpath=db_fp, train_size=TRAIN_SIZE,
                                           patch_size=PATCH_SIZE, std_thrsh=STD_THRESH)
    #
    # Normalize patches
    train_data -= np.mean(train_data, axis=0)
    train_data /= np.std(train_data,  axis=0)
    #
    #learn dictionary 
    train_dict = Traindict(max_iter=MAX_ITER, alpha=ALPHA, atom_count=100, step_size=10)
    Wd, loss   = train_dict.learn_Wd_gd(train_data.T)
    #
    #save learn dictionary
    learned_dict_fp = os.path.dirname(os.path.realpath(__file__)) + \
    '\\..\\learned_dict\\Wd_norm_iter{}_alpha{}_stdthrsh{}.py'.format(MAX_ITER, ALPHA, STD_THRESH)
    np.save(learned_dict_fp, Wd)
    display_atoms(Wd.T, PATCH_SIZE)

    plt.plot(loss)
    plt.ylabel('loss')
    plt.title('loss per iteration')
    plt.show()

    


