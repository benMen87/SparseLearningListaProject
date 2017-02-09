import cod
import numpy as np
import scipy.io as scio
from sklearn import decomposition
from sklearn import linear_model

Wd = scio.loadmat('m_ref/Wd.mat')
Wd = Wd['Wd']
X = scio.loadmat('m_ref/X.mat')
X = X['X']

cc = cod.CoD(Wd)

X_norm = X / np.linalg.norm(X)
Z_norm = cc.run_cod(X_norm)

reg = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
ZZ = reg.fit(Wd, X_norm)