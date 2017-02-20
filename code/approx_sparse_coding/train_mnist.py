import mnist
#import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sparse_coding import cod, ista
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn import linear_model
import lcod

traindata = np.load(r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\mnist\train.npy')
input     = traindata[:, 2:]
label     = np.zeros(shape=(input.shape[0], 10))
label[np.arange(traindata[:, 2:].shape[0]), traindata[:, 1].astype(np.int)] = 1 

dico = MiniBatchDictionaryLearning(n_components=400, alpha=0.5, n_iter=1000)
Wd = dico.fit(input).components_.T

clf = linear_model.Lasso(alpha=0.1)
    
model = lcod.LCoD(We_shape=Wd.T.shape, unroll_count=7)





