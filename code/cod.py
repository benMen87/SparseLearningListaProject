import numpy as np
import logging



class CoD(object):
    """
    Learn sprse represintainon of input
    """

    def __init__(self, Wd,  max_iter=1000, thresh=1e-4, alpha=0.1, verbose=False):
        self.max_iter = max_iter
        self.thresh   = thresh
        self.alpha    = alpha
        self.Wd       = Wd
        self.loss     = lambda Wd, X,Z: 0.5*np.linalg.norm(X - np.matmul(Wd, Z))**2 + self.alpha*np.linalg.norm(Z, 1) 
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def soft_threshold(self, x, threshold):
        x = x.copy()
        j = np.abs(x) <= threshold
        x[j] = 0
        x  = x - np.sign(x)*threshold
        return x

    def run_cod(self, X, Wd=None, max_iter=None, thresh=None, alpha=None):
        """
        Find sparse representation via coordiant dicent.
        """
        Wd = self.Wd if Wd is None else Wd
        max_iter = self.max_iter if max_iter is None else max_iter
        thresh   = self.thresh   if thresh is None else thresh
        alpha   =  self.alpha    if alpha is None else alpha

        if np.ndim(X) == 1:
            X = X[:, np.newaxis]

        n, m  = Wd.shape
        # S = I - Wd*Wd^T
        S = np.eye(m) - np.einsum('ji, jk->ik', Wd, Wd)
        # B = Wd^T*X
        B = np.einsum('ij, ik->jk', Wd, X)
        Z    = np.zeros((m, 1))

        for i in range(0, max_iter):
            logging.debug('iter %d'%i)
            logging.debug('loss: %f' %self.loss(Wd, X, Z))
            Zhat = self.soft_threshold(B, alpha)
            res  = Zhat - Z
            k    = np.argmax(np.abs(res), axis=0)
            B    = B + S[:,k]*res[k]
            Z[k] = Zhat[k]
            if np.linalg.norm(res , 2) < thresh : 
                break
        else:
            logging.debug('sparse rep did not converge broke after max iter')
        Z = self.soft_threshold(B, alpha)

        return Z
