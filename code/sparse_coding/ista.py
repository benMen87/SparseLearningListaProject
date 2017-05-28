import numpy as np
import logging
import scipy


class ISTA(object):
    """
    Learn sparse representation of input
    """
    def __init__(self, Wd, L=None, max_iter=1000, thresh=1e-4, alpha=0.5, verbose=True):
        self.max_iter = max_iter
        self.thresh = thresh
        self.alpha = alpha
        self.Wd = Wd
        if L is None:
            L = max(abs(np.linalg.eigvals(np.matmul(Wd.T, Wd))))
        self.L = L
        
        self.loss = lambda Wd, X, Z: 0.5*np.linalg.norm(X - np.matmul(Wd, Z))**2 + self.alpha*np.linalg.norm(Z, 1) 

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def soft_threshold(self, x, threshold):
        x = x.copy()
        j = np.abs(x) <= threshold
        x[j] = 0
        x  = x - np.sign(x)*threshold
        return x

    def fit(self, X, Wd=None, L=None, max_iter=None, thresh=None, alpha=None):
        """
        Find sparse representation via ista alg.
        """

        Wd = self.Wd if Wd is None else Wd
        L = self.L if L is None else L
        max_iter = self.max_iter if max_iter is None else max_iter
        thresh = self.thresh if thresh is None else thresh
        alpha = self.alpha if alpha is None else alpha

        if np.ndim(X) == 1:
            X = X[:, np.newaxis]
        n, m = Wd.shape

        mini_batch_size = X.shape[1] 
        Z = np.zeros((m, mini_batch_size))
        S = np.eye(m) - (1/L)*np.matmul(Wd.T, Wd)
        B = (1/L)*np.matmul(Wd.T, X)
        theta = alpha / L

        loss_arr = []
        for i in range(0, max_iter):
            loss_arr.append(self.loss(Wd, X, Z))

            C = np.matmul(S, Z) + B
            Zhat = self.soft_threshold(C, theta)
            res = Zhat - Z
            Z = Zhat

            if np.linalg.norm(res, 2) / mini_batch_size < thresh : 
                break
        else:
            logging.debug('sparse rep did not converge broke after max iter')
        return Z, loss_arr