
# ------------------------ IMPORT DEPEDINCYS -------------------------

# %%

import tensorflow as tf
import shutil
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
import argparse

import lmnist
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn import linear_model
from train import test as sparse_code_test
from sklearn.manifold import TSNE
from mnist import MNIST
from lcod import LCoD
from lista import LISTA
from lista_convdict2d import LISTAConvDict2d

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + '/..')
from dict_learning import traindict

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-n', '--number_epoch', default=1, type=int)
parser.add_argument('-e', '--eta', type=float, default=0.1,
                    help='loss = eta*loss_SC+(1-eta)loss_MNIST')
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float)
parser.add_argument('-m', '--model', default='lista', help='lcod/lista')
parser.add_argument('-u', '--unroll_count', default=3, type=int)
parser.add_argument('-w', '--warm_restart', type=bool, default=True)

args = parser.parse_args()

MNIST_SET_PATH = DIR_PATH + '/../../mnist_data/'

try:
    mndata = MNIST(MNIST_SET_PATH)
except:
    raise Exception('please dowlod mnist data from http://yann.lecun.com/exdb/mnist/ \
                     to {}/../../../mnist_data/'.format(DIR_PATH))

# --------------------------------- LEARN DICTIOARY ---------------------------

# %%


def load_maybe_build_dictionary(dict_path, atom=784):
    if not os.path.exists(dict_path):
        print('Learning new dict this may take a while')
        input, _ = mndata.load_training()
        input -= np.mean(input, axis=1, keepdims=True)
        input /= np.linalg.norm(input, axis=1, keepdims=True)
        dico = MiniBatchDictionaryLearning(n_components=atom, alpha=0.1,
                                           n_iter=1000)
        Wd = dico.fit(input[:6000]).components_.T
        np.save(dict_path, Wd)
    else:
        Wd = np.load(dict_path)
    return Wd

Wd = load_maybe_build_dictionary(DIR_PATH + '/mnist_data/conv2d/Wd.npy', atom=784*6)

# ----------------------------- GET TRAIN/TEST DATA ---------------------------

# %%


def load_maybe_build_sc(input, sc_path):

    if not os.path.exists(sc_path):
        print('{} does not exist.\n Building new clean data.\
               This will take a while...'.format(sc_path))

        input -= np.mean(input, axis=1, keepdims=True)
        input /= np.linalg.norm(input, axis=1, keepdims=True)

        sparse_coder = cod.CoD(Wd, alpha=0.5)
        sc_arr = []
        for X in input:
            Z, _ = sparse_coder.fit(X)
            sc_arr.append(Z)
        np.savez(sc_path, Z=sc_arr)
    else:
        data = np.load(sc_path)
        sc_arr = data['Z']
    return sc_arr

train_im, train_label = mndata.load_training()
train_sparse = load_maybe_build_sc(train_im, MNIST_SET_PATH + '/conv2d/' +  'train_sc.npz')

train_im -= np.mean(train_im, axis=1, keepdims=True)
train_im /= np.linalg.norm(train_im, axis=1, keepdims=True)

test_im, test_label = mndata.load_testing()
test_sparse = load_maybe_build_sc(test_im, MNIST_SET_PATH + '/conv2d/' + 'test_sc.npz')

test_im -= np.mean(test_im, axis=1, keepdims=True)
test_im /= np.linalg.norm(test_im, axis=1, keepdims=True)
# ---------------------- MINIBATCH GENETRATORS ----------------

# %%


class Batchgen():

    def __init__(self, img, sc=None, lbl=None, batch_size=1, max_cycles=np.inf):
        self._img = img
        self._sc = sc
        self._lbl = lbl
        self._batch_size = batch_size
        self._curr_pos = 0
        self._overlap = 0
        self._max_cycles = max_cycles

    def __iter__(self):
        return self

    def next(self):
        if self._max_cycles < self._curr_pos % len(self._img):
            raise StopIteration
        else:
            return self.next_batch()

    def next_batch(self):
        b_img = []
        b_sc = []
        b_lbl = []

        ret_list = [b_img]

        for i in range(self._batch_size):
            pos = (self._curr_pos + i) % len(self._img)
            b_img.append(self._img[pos - self._overlap])
            if self._sc is not None:
                b_sc.append(np.squeeze(self._sc[pos - self._overlap]))
            if self._lbl is not None:
                label = np.zeros(10)
                label[self._lbl[pos - self._overlap]] = 1
                b_lbl.append(label)
        self._curr_pos += self._batch_size

        if len(b_sc):
            ret_list.append(b_sc)
        if len(b_lbl):
            ret_list.append(b_lbl)
        return tuple(ret_list)


batch_size = args.batch_size
valid_size = np.int(len(train_im) * 0.1)

validset = Batchgen(train_im[:valid_size],
                    train_sparse[:valid_size],
                    train_label[:valid_size], batch_size=1)

trainset = Batchgen(train_im[valid_size:],
                    train_sparse[valid_size:],
                    train_label[valid_size:], batch_size=batch_size)
testset = Batchgen(img=test_im, lbl=test_label, batch_size=1, max_cycles=6000)
# --------------------------------- helper funcs -----------------------------
# %%


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def messure_calssify_accuracy(data_gen, sess, model):

    predictions = []
    labels = []
    iter = 0
    for im, label in data_gen:

        pred, Z, l_Z = sess.run([model.predict, model.input,
                                 model._sc_block.output],
                                feed_dict={model.input: im,
                                           model.labels: label})
        predictions.append(np.squeeze(pred.T))
        labels.append(np.squeeze(np.array(label).T))
        iter += 1
        if iter % 50 == 0:
            print('test accuracy iter %d' % iter)
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

    # plt.ion()
    plt.show()

# -------------------------------- Build Graph ------------------------------

# %%

# g = tf.Graph()
# with g.as_default():
if args.warm_restart:
    We = Wd.T
else:
    We = None

if args.model == 'lcod':
    sc_block = LCoD(We_shape, unroll_count, We,
                    shrinkge_type='soft thresh',
                    batch_size=batch_size)
elif args.model == 'lista':
    sc_block = LISTA(We_shape, unroll_count, We,
                     shrinkge_type='soft thresh',
                     batch_size=batch_size)
elif args.model == 'lista_convdict2d':
    kernel_count = We.shape[0] // We.shape[1]
    kernael_size = 3

    filter_arr = [np.random.randn(kernel_size, kernel_size)
                  for _ in range(kernel_count)]
    filter_arr = np.array([f/np.linalg.norm(f) for f in filter_arr])
    We_shape = (len(filter_arr)*We.shape[1], We.shape[1])
    sc_block = LISTAConvDict2d(We_shape=We.shape,
                               unroll_count=unroll_count,
                               filter_arr=filter_arr, L=L,
                               batch_size=batch_size,
                               kernal_size=kernel_size,
                               init_params_dict=dict())

model = lmnist.Lmnist(We_shape=We.shape, unroll_count=args.unroll_count,
                      sc_block=sc_block, We=We, sc_type=args.model,
                      batch_size=batch_size)
model.build_model()
#
# optimize graph with gradient decent with LR of 1/t
global_step = tf.Variable(0, trainable=False)
learning_rate = args.learning_rate
k = 0.5
decay_rate = 1
learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k, decay_rate)

eta = args.eta
total_loss = eta*model.sparse_loss + (1-eta)*model.classify_loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# --------------------------------- TRAINING ---------------------------------

# %%

print(30*'='+'Restart' + 30*'=')

train_loss = []
vld_loss = []
vld_clssloss = []
vld_scloss = []
vld_sparsity = []
number_of_steps = args.number_epoch
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
print('Initialized')
for step in range(number_of_steps):
    img, Z_star, label = trainset.next_batch()
    _, totloss, pred = sess.run([optimizer, total_loss,
                                 model.predict],
                                {model.input: img,
                                 model.sparse_target: Z_star,
                                 model.labels: label})
    train_loss.append(totloss)
    if step % 5000 == 0:
        print("step %d loss: %f," % (step, totloss))
        print("predict: {}, label: {}".format(np.argmax(pred, axis=-1),
                                              np.argmax(label, axis=-1)))
    if (step % 1000) == 0:
        tot = 0
        cls = 0
        sc = 0
        sparsity = 0
        for _ in range(valid_size):
            img_v, Z_v, label_v = validset.next_batch()
            feed_dict = {model.input: img_v,
                         model.sparse_target: Z_v,
                         model.labels: label_v}
            totloss, sc_loss, clss_loss, Z = sess.run([total_loss,
                                                      model.sparse_loss,
                                                      model.classify_loss,
                                                      model.Z],
                                                      feed_dict=feed_dict)
            tot += totloss
            cls += clss_loss
            sc += sc_loss
            sparsity += np.count_nonzero(Z)
        vld_loss.append(tot / valid_size)
        vld_clssloss.append(cls / valid_size)
        vld_scloss.append(sc / valid_size)
        vld_sparsity.append(sparsity / valid_size)
        print('Valid run tot loss: %f class loss: %f,\
               SC loss: %f, sprasity: %f' %
              (vld_loss[-1], vld_clssloss[-1], vld_scloss[-1], vld_sparsity[-1]))
# -------------------------- TEST MODEL + PLOTS -----------------------------
# %%
log_dir = DIR_PATH + '/logdir'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_res_dir = log_dir + '/data_result'
if not os.path.exists(data_res_dir):
    os.makedirs(data_res_dir)
data_res_dir += '/lmnist'
if not os.path.exists(data_res_dir):
    os.makedirs(data_res_dir)
plot_dir = log_dir + '/plots'
if not os.path.exists(data_res_dir):
    os.makedirs(data_res_dir)
plot_dir += '/lmnist'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


Z_TSNE = []
labels_TSNE = []
for i in range(400):
    img, _,  label = validset.next_batch()
    Z = sess.run(model.Z, feed_dict={model.input: img})
    Z_TSNE.append(np.reshape(Z, 784))
    labels_TSNE.append(np.argmax(label))
np.savez(data_res_dir + '/final_embbed', Z=Z_TSNE, L=labels_TSNE)

print('='*50+'TEST RESULTS'+'='*50)
#
# test classification performance
acc = messure_calssify_accuracy(testset, sess, model)
print('Classification test accuracy: %f' % acc)
#
# test SC performance
approx_err, sc_err = sparse_code_test(sess, model.sparsecode_block,
                                      Batchgen(img=test_im, sc=test_sparse, max_cycles=6000), 3, Wd,
                                      'lista', 50)
print('Lcod err: %.6f, Cod err: %.6f' % (approx_err, sc_err))

plt.figure(1)
plt.subplot(211)
plt.plot(train_loss[1:])
plt.ylabel(' train loss')
plt.subplot(212)
plt.plot(vld_loss[1:])
plt.ylabel('valid loss')
plt.savefig(plot_dir + '/total_loss_{}'.format(model.sparsecode_block.unroll_count),
            bbox_inches='tight')

plt.figure(2)
plt.subplot(311)
plt.plot(vld_clssloss[1:])
plt.ylabel('validation classification loss')
plt.subplot(312)
plt.plot(vld_scloss[1:])
plt.ylabel('validation SC loss')
plt.subplot(313)
plt.plot(vld_sparsity[1:])
plt.ylabel('validation sparsity loss')
plt.savefig(plot_dir + '/sc_cls_split_loss_{}'.format(model.sparsecode_block.unroll_count),
            bbox_inches='tight')

# --------------- Plot TSNE SC embeddings ---------------------------

# %%


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2),
                       textcoords='offset points', ha='right', va='bottom')
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(Z_TSNE)
# labels = ['{}'.format(np.argmax(labels_TSNE[i])) for i in range(50)]
labels = labels_TSNE
plot(two_d_embeddings, labels)

plt.show()