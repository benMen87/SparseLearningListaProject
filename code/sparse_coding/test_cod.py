import cod
import numpy as np
import scipy.io as scio
from sklearn import decomposition
from sklearn import linear_model
import matplotlib.pyplot as plt
import os

#Wd = scio.loadmat('m_ref/Wd.mat')
#Wd = Wd['Wd']
#X = scio.loadmat('m_ref/X.mat')
#X = X['X']

Wd       = np.random.randn(10 ** 2, 100)
col_norm = np.linalg.norm(Wd, axis=0)
Wd       = Wd / col_norm


import db_tools


db_fp = os.path.dirname(os.path.realpath(__file__)) + '\\..\\images\\BSDS300-images.tgz' 

train_data = db_tools.next_patch_gen(db_fp, (10, 10), 1.0)
im = next(train_data)
X = im.reshape(100, 1)

cc = cod.CoD(Wd)

X_norm = X - np.mean(X)
X_norm /= np.linalg.norm(X_norm)

Z_norm, loss = cc.run_cod(X_norm)

print('Z is {} sparse out of {} elements\n'.format(np.count_nonzero(Z_norm), len(Z_norm)))
print('reconstruction error: {}'.format(np.linalg.norm(X_norm - np.matmul(Wd, Z_norm))))

plt.plot(loss)
plt.ylabel('loss')
plt.title('loss per iteration')
plt.show()


reg = linear_model.Lasso(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
ZZ = reg.fit(Wd, X_norm)