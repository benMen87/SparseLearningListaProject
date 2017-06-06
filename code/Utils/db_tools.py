from sklearn.feature_extraction import image
import tarfile as tar
import numpy as np
from PIL import Image
import os
import sys
from collections import namedtuple
from six.moves import cPickle
import gzip

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from sparse_coding.cod import CoD
from sparse_coding.ista import ISTA

def next_image_gen(db_fullpath, dset_typ='train', rgb2gray=True):

    db_p = tar.open(db_fullpath)
    for f_info in db_p.getmembers():
        if dset_typ in f_info.name and f_info.name.endswith('.jpg'):
            img_fp = db_p.extractfile(f_info)
            I = Image.open(img_fp)
            if rgb2gray:
                I = np.asarray(I.convert('L'))
            yield I


def next_patch_gen(db_fullpath, patch_size, std_thrsh, dset_typ='train',  max_ppi=np.inf):
    """
    Generator for iterating over patches of training set images.
    INPUT:
    db_fullpath - full path to full data base expected in tar format
    patch_size  -  tuple of patch size (h, w)
    std_thrsh   - discard patches with std lower than threshold
    max_ppi     - max amount of patches per image 
    """
    img_iter = next_image_gen(db_fullpath, dset_typ)
    for im in img_iter:
        patches = image.extract_patches_2d(im, patch_size)
        patches_skipped = 0
        for p_num, p in enumerate(patches):
            if p_num - patches_skipped > max_ppi:
                break
            if np.std(patches) < std_thrsh:
                patches_skipped += 1
                continue
            yield p


def load_train_data_to_mem(db_fullpath, patch_size, std_thrsh, train_size, dset_typ='train'):

    train_data = []
    patch_iter = next_patch_gen(db_fullpath, patch_size, std_thrsh, dset_typ)

    for i in range(np.uint64(train_size)):
        print('patch number %d out of %d'%(i, train_size))
        p = next(patch_iter)
        train_data.append(p.reshape(patch_size[0]*patch_size[1])) #collum stack patch
    return np.asarray(train_data)


def load_maybe_build_train_set(train_fullpath, db_fullpath=None, train_size=None,
                               patch_size=None, std_thrsh=None, dset_typ='train', savefile=False):
        """
        Load training data if it exist if not run through image db and build patch
        training data.
        INPUT: db_fullpath    - full path to tar database
               train_fullpath - full path to training/test patches if exists otherwise '' can be passed
               patch_size    - tuple i.e. (h, w)
                std_thrsh    -  is std(patch) < std_thrsh discard patch
        """
        #
        # We may have some or all the patches we need saved in correct format
        saved_train_data = np.empty(shape=(0, patch_size[0]*patch_size[1]))
        # if exists load it
        if os.path.isfile(train_fullpath):
            try:
                saved_train_data = np.load(train_fullpath)
                train_size -= saved_train_data.shape[0]
                if train_size <= 0:
                    return saved_train_data
            except:
                print('Error when loading train-set will try to rebuild train set')
        # else build it
        train_data = load_train_data_to_mem(db_fullpath, patch_size, std_thrsh,
                                            train_size, dset_typ)
        train_data = np.append(train_data, saved_train_data, axis=0)
        if savefile:
            np.save(train_fullpath, train_data)
        return train_data

##############################################################
#                   Tools for Lista/LCoD                     #
##############################################################


def load_dictioary(dict_path):
    try:
        Wd = np.load(dict_path)
        return Wd
    except:
        raise FileNotFoundError('Cannot find dictionary in specified path %s'%dict_path)


def load_data_set(dataset_path):
    try:
        data_set = np.load(dataset_path)
        return data_set
    except:
        raise FileNotFoundError('Cannot find dictionary in specified path %s'%dataset_path)


def load_dataset_and_dict(datapath, dictpath):

    Wd = load_dictioary(dictpath)
    patches = load_data_set(datapath)

    set_size, patch_size = patches.shape
    atom_size, atom_count = Wd.shape

    if not patch_size == atom_size:
        raise ValueError('There is a mismatch between Dictionary and data set\
                         the length of the diction and the size of \
                         the patches should be equal. Also use the dictionary\
                         that was learned over this data set.')
    return Wd, patches


def compute_patch_sc_pydict(patch_set, Wd):
    sparse_rep = []
    sparse_code = ISTA(Wd, alpha=0.5)
    i = 0
    for patch in patch_set:
        i += 1
        if i % 100 == 0:
            print('patch number %d'%i)
        Z, _ = sparse_code.fit(X=patch)
        sparse_rep.append(Z)
    sparse_rep = np.asarray(sparse_rep)
    return {'X': patch_set, 'Y': sparse_rep}


def basic_X_Z_gen(input, labels, run_once=False):
    while True:
        for (X, Z) in zip(input, labels):
            yield (X, Z)
        else:
            if run_once:
                break


def batch_X_Z_gen(input, labels, batch_size=1, run_once=False):
    basic_gen = basic_X_Z_gen(input, labels, run_once)

    batch_X = []
    batch_Z = []

    for X, Z in basic_gen:

        if np.ndim(X) > 1:
            X = np.squeeze(X)
        if np.ndim(Z) > 1:
            Z = np.squeeze(Z)

        batch_X.append(X)
        batch_Z.append(Z)

        if len(batch_X) == batch_size:
            yield batch_X, batch_Z
            batch_X = []
            batch_Z = []


def testset_gen(data_test_path, run_once=True):
    dt = np.load(data_test_path)

    input = dt['X']
    labels = dt['Y']

    input -= np.mean(input, axis=1, keepdims=True)
    input /= np.std(input, axis=1, keepdims=True)

    return basic_X_Z_gen(input, labels, True)


def trainset_gen(data_train_path, valid_size=500, batch_size=1):
    dt = np.load(data_train_path)

    input = dt['X']
    labels = dt['Y']

    input -= np.mean(input, axis=1, keepdims=True)
    input /= np.std(input, axis=1, keepdims=True)
    #
    # random shuffle
    permutation = np.random.permutation(labels.shape[0])
    permutation = np.random.permutation(labels.shape[0])

    input = input[permutation, :]
    labels = labels[permutation, :]

    trainset_gen = batch_X_Z_gen(input[valid_size:], labels[valid_size:],
                                 batch_size=batch_size)
    validset_gen = batch_X_Z_gen(input[:valid_size], labels[:valid_size])

    Datagens = namedtuple('Generators', 'train_gen valid_gen')
    if valid_size != 0:
        ret_gens = Datagens(train_gen=trainset_gen, valid_gen=validset_gen)
    else:
        ret_gens = Datagens(train_gen=trainset_gen, valid_gen=None)
    return ret_gens


def build_approx_sc_learnig_data(data_train_path, data_test_path,
                                 dict_train_path, dict_test_path, outpath,
                                 train_size=100000, test_size=30000):
    """
    """
    #
    # training data
    if not os.path.isfile(outpath + '/trainset.npz'):
        print('building new trainset')
        Wd, patches = load_dataset_and_dict(datapath=data_train_path,
                                            dictpath=dict_train_path)
        patches -= np.mean(patches, axis=1, keepdims=True)
        patches /= np.std(patches, axis=1, keepdims=True)

        data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
        np.savez(outpath + '/trainset', X=data_lable_dict['X'], Y=data_lable_dict['Y'])
    else:
        print('trainset: {} \n exists already please rename or change dir if \
              you wish to create a new one'.format(outpath + '/trainset.npy'))
    #
    # test data
    if not data_test_path == '' and \
       not os.path.isfile(outpath + '/testset.npz'):
        Wd, patches = load_dataset_and_dict(datapath=data_test_path,
                                            dictpath=dict_test_path)
        patches -= np.mean(patches, axis=1, keepdims=True)
        patches /= np.std(patches, axis=1, keepdims=True)

        data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
        np.savez(outpath + '/testset', X=data_lable_dict['X'], Y=data_lable_dict['Y'])
