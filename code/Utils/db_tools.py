from sklearn.feature_extraction import image
import tarfile as tar
import numpy as np
from PIL import Image
import os
import sys
from collections import namedtuple

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from sparse_coding.cod import CoD

def next_image_gen(db_fullpath, rgb2gray=True):

     db_p = tar.open(db_fullpath)
     for f_info in db_p.getmembers():
         if 'train' in f_info.name and f_info.name.endswith('.jpg'):
             img_fp = db_p.extractfile(f_info)
             I = Image.open(img_fp)
             if rgb2gray:
                 I = np.asarray(I.convert('L'))
             yield I

def next_patch_gen(db_fullpath, patch_size, std_thrsh, max_ppi=np.inf):
    """
    Generator for iterating over patches of training set images.
    INPUT:
    db_fullpath - full path to full data base expected in tar format
    patch_size  -  tuple of patch size (h, w)
    std_thrsh   - discard patches with std lower than threshold
    max_ppi     - max amount of patches per image 
    """
    img_iter = next_image_gen(db_fullpath)
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

def load_train_data_to_mem(db_fullpath, patch_size, std_thrsh, train_size):

    train_data = []
    patch_iter = next_patch_gen(db_fullpath, patch_size, std_thrsh)

    for i in range(np.uint64(train_size)):
        print('patch number %d out of %d'%(i, train_size))
        p = next(patch_iter)
        train_data.append(p.reshape(patch_size[0]*patch_size[1])) #collum stack patch
    return np.asarray(train_data)

def load_maybe_build_train_set(train_fullpath, db_fullpath=None, train_size=None,
                               patch_size=None, std_thrsh=None, savefile=False):
        """
        Load trainig data if it exist if not run through image db and build patch
        trainig data.
        INPUT: db_fullpath    - full path to tar database
               train_fullpath - full path to trainig patches if exists if not where it should be saved can pass ''
               patch_size    - tuple i.e. (h, w)
                std_thrsh    -  is std(patch) < std_thrsh discard patch
        """
        #
        # We may have some or all the patches we need saved in correct format
        saved_train_data = np.empty(shape=(0, patch_size[0]*patch_size[1]))
        #if exists load it
        if os.path.isfile(train_fullpath):
            try: 
                saved_train_data = np.load(train_fullpath)
                train_size -= saved_train_data.shape[0]
                if train_size <= 0:
                    return saved_train_data
            except:
                print('Error when loading train-set will try to rebuild train set') 
        # else build it
        train_data = load_train_data_to_mem(db_fullpath, patch_size, std_thrsh, train_size)
        train_data = np.append(train_data, saved_train_data, axis=0)
        if savefile:
            np.save(train_fullpath, train_data)
        
        return train_data

##############################################################
#                   Tools for Lista/Lcod                     #
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

    Wd   = load_dictioary(dictpath)
    patches = load_data_set(datapath)

    set_size, patch_size  = patches.shape
    atom_size, atom_count = Wd.shape
     
    if not patch_size == atom_size:
        raise ValueError('There is a mismatch between Dictionary and data set the length of the diction and the size of \
        the patches should be equal. Also use the dictionary that was learned over this data set.')

    return Wd, patches

def compute_patch_sc_pydict(patch_set, Wd):
    sparse_rep = []
    sparse_code = CoD(Wd, alpha=0.5)

    for patch in patch_set:
        Z, _ = sparse_code.run_cod(X=patch)
        sparse_rep.append(Z)
    sparse_rep = np.asarray(sparse_rep)
    return {'X':patch_set, 'Y':sparse_rep}


def trainset_gen(data_train_path, valid_ratio=0.2):
    dt = np.load(data_train_path)
    data_dict = dt.item()

    train_offset = int(valid_ratio*len(data_dict['X']))

    input   = data_dict['X']
    labels  = data_dict['Y'] 

    input -= np.mean(input, axis=1, keepdims=True)
    input /= np.std(input, axis=1, keepdims=True)
    #
    # random shuffle
    permutation = np.random.permutation(labels.shape[0])
    input = input[permutation, :]
    labels = labels[permutation, :]
    def trainset_gen():
        while True:
            for (X, Z) in zip(input[train_offset:], labels[train_offset:]):
                if np.ndim(X) == 1:
                    X = X[:, np.newaxis]
                if np.ndim(Z) == 1:
                    Z = Z[:, np.newaxis]
                yield (X, Z)

    def validset_gen():
        while True:
            for (X, Z) in zip(input[:train_offset], labels[:train_offset]):
                if np.ndim(X) == 1:
                    X = X[:, np.newaxis]
                if np.ndim(Z) == 1:
                    Z = Z[:, np.newaxis]
                yield (X, Z)

    Datagens= namedtuple('Generators', 'train_gen valid_gen')
    if valid_ratio != 0:
        ret_gens =  Datagens(train_gen=trainset_gen(), valid_gen=validset_gen())
    else:
        ret_gens =  Datagens(train_gen=trainset_gen(), valid_gen=None)
    return ret_gens 


def build_approx_sc_learnig_data(data_train_path, data_test_path, dict_train_path, dict_test_path, outpath): 
    """
    """
    #training data
    Wd, patches = load_dataset_and_dict(datapath=data_train_path, dictpath=dict_train_path)

    patches -= np.mean(patches, axis=1, keepdims=True)
    patches /= np.std(patches, axis=1, keepdims=True)

    data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
    np.save(outpath + '/trainset.npy', data_lable_dict)

    #test data
    if not data_test_path == '':
        Wd, patches = load_dataset_and_dict(datapath=data_test_path, dictpath=dict_test_path)
        data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
        np.save(outpath + '\testset.npy', data_lable_dict)