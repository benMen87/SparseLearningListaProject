import lmnist
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sparse_coding import cod, ista
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn import linear_model
from train import zero_none_grad
from train import test as sparse_code_test
from sklearn.manifold import TSNE
from matplotlib import pylab
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from dict_learning import traindict

#
######################### LEARN DICTIOARY #############################################
dir_path = os.path.dirname(os.path.realpath(__file__))
dict_path = dir_path + '/mnist_data/Wd.npy' 
if not os.path.exists(dict_path):
    dico = MiniBatchDictionaryLearning(n_components=784, alpha=0.1, n_iter=1000)
    Wd = dico.fit(input).components_.T
    np.save(dict_path, Wd)
else:
    Wd = np.load(dict_path)
#traindict.display_atoms(Wd.T, (28, 28))

##################################### GET TRAIN/TEST DATA #######################################
def load_maybe_build_data(raw_data_path, clean_data_path):

    if not os.path.exists(clean_data_path):
        data = np.load(raw_data_path)
        input     = data[:, 1:]
        label     = np.zeros(shape=(input.shape[0], 10))
        label[np.arange(data.shape[0]), data[:, 0].astype(np.int)] = 1 

        input -= np.mean(input, axis=1, keepdims=True)
        input /= np.linalg.norm(input, axis=1, keepdims=True)
        
        sparse_coder = cod.CoD(Wd, alpha=0.5)
        Z_arr = []
        X_arr = []
        for X in input:
            Z, _ = sparse_coder.fit(X)
            X_arr.append(X)
            Z_arr.append(Z)
        np.savez(clean_data_path, X=X_arr,Z=Z_arr, L=label)
        X_arr = np.array(X_arr)
        Z_arr = np.array(Z_arr) 
    else:
         data = np.load(clean_data_path)
         X_arr = data['X']
         Z_arr = data['Z']
         L_arr = data['L']

    permutation = np.random.permutation(X_arr.shape[0])
    X_arr = X_arr[permutation, :]
    Z_arr = Z_arr[permutation, :]
    L_arr = L_arr[permutation, :]

    return X_arr, Z_arr, L_arr


def data_gen(iter_type, one_sweep=False):
    while True:
        for el in iter_type:
            yield el
        if one_sweep:
            break


raw_train_data = dir_path + '/../../mnist/train.npy'
train_path = dir_path + '/mnist_data/train.npz'
train_im, train_sparse, train_label = load_maybe_build_data(raw_train_data, train_path)
valid_size = np.int(train_im.shape[0] * 0.1)
validset = list(zip(train_im[:valid_size], train_sparse[:valid_size], train_label[:valid_size]))
trainset_gen = data_gen(list(zip(train_im[valid_size:], train_sparse[valid_size:], train_label[valid_size:])))

raw_test_data = dir_path + '/../../mnist/test.npy'
test_path = dir_path + '/mnist_data/test.npz'
test_im, test_sparse, test_label = load_maybe_build_data(raw_test_data, test_path)
#testset = zip(test_im, test_sparse, test_label)

######################################## helper funcs #################################
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def messure_calssify_accuracy(data_gen, sess, model):
    predictions = []
    labels      = []
    iter = 0
    for im, label in data_gen:
        pred = sess.run([model.predict], feed_dict={model.input: im, model.labels: label})
        predictions.append(pred)
        labels.append(label)
        iter+=1
        if iter%50 == 0:
            print('test accuracy iter %d'%iter)
    acc = accuracy(np.array(predictions), np.array(labels))
    return acc

def display_atoms(Wd, patch_size):

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(Wd[:]):
        plt.subplot(patch_size[0], patch_size[1], i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
           interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    #plt.ion()
    plt.show()

####################################### Build Graph ###################################
g = tf.Graph()
with g.as_default():
    We = Wd.T    
    model = lmnist.Lmnist(We_shape=We.shape, unroll_count=3, We=We)
    
    model.build_model()
    
    #
    # optimize graph with gradient decent with LR of 1/t
    global_step   = tf.Variable(0, trainable=False)
    learning_rate = 0.005
    k             = 0.5
    decay_rate    = 1
    learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k, decay_rate)
        
    #
    # Clip gradients to avoid overflow due to recurrent nature of algorithm
    eta = 0.2
    optimizer  = tf.train.GradientDescentOptimizer(learning_rate) 
    gvs_sc        = optimizer.compute_gradients(eta*model.sparse_loss)
    capped_sc_gvs = [(tf.clip_by_value(zero_none_grad(grad, var), -1, 1), var) for grad, var in gvs_sc]
    gvs_clss      = optimizer.compute_gradients((1-eta)*model.classify_loss)
    optimizer  = optimizer.apply_gradients(gvs_clss + gvs_sc)
    total_loss = eta*model.sparse_loss + (1-eta)*model.classify_loss

########################################## TRAINING ###################################################
print(30*'='+'Restart' + 30*'=')

train_loss    = []
vld_loss      = []
vld_clssloss  = []
vld_scloss    = []
number_of_steps = 3000
with  tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(number_of_steps):
        img, Z_star, label = next(trainset_gen)
         
        Z_star = np.reshape(Z_star, (Z_star.shape[0]))
        _, totloss, pred, l_We = sess.run([optimizer, total_loss, model.predict, model._locd.We],
                                 {model.input: img, model.sparse_target: Z_star, model.labels: label})
        train_loss.append(totloss)
        if (step % 100)==0:
            print("step %d loss: %f"%(step, totloss))

        if (step % 1000) == 0:
            tot=0;cls=0;sc=0
            for img_v, Z_v, label_v in  validset:
                feed_dict = {model.input: img, model.sparse_target:Z_star, model.labels: label}
                totloss, sc_loss, clss_loss = sess.run([total_loss, model.sparse_loss, model.classify_loss],\
                                                        feed_dict=feed_dict)
                tot += totloss
                cls += clss_loss
                sc  += sc_loss

            vld_loss.append(tot /  valid_size)     
            vld_clssloss.append(cls / valid_size)  
            vld_scloss.append(sc / valid_size)
            print('Valid run tot loss: %f class loss: %f, SC loss: %f'%(vld_loss[-1], vld_clssloss[-1], vld_scloss[-1]))

    print('='*50+'TEST RESULTS'+'='*50)
    #
    # test classification performance
    acc = messure_calssify_accuracy(zip(test_im, test_label), sess, model)
    print('Classification test accuracy: %f' %acc)
    #
    # test SC performance
    approx_err, sc_err, Z_lista = sparse_code_test(sess, model._locd, zip(test_im, test_sparse), 3,\
                                  Wd, 'cod', test_im.shape[0])
    print('Lcod err: %.6f, Cod err: %.6f' % (approx_err, sc_err))

plt.figure(1)
plt.subplot(211)
plt.plot(train_loss)
plt.ylabel(' train loss') 
plt.subplot(212)
plt.plot(vld_loss)
plt.ylabel('valid loss')

plt.figure(2)
plt.subplot(211)
plt.plot(vld_clssloss)
plt.ylabel('classification loss')
plt.subplot(212)
plt.plot(vld_scloss)
plt.ylabel('SC loss')

display_atoms(l_We, (28, 28))

try:
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(Z_lista[1:num_points+1, :])
    
    def plot(embeddings, labels):
      assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
      pylab.figure(figsize=(15,15))  # in inches
      for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    
    numbers = ['{}'.format(np.argmax(test_label[i])) for i in range(1, num_points+1)]
except:
    print('Somthing went wrog with TNSE')
    print("Unexpected error:", sys.exc_info()[0])

plt.show()