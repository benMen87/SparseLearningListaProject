import numpy as np
import os
import sys
from collections import namedtuple

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from sparse_coding.cod import CoD


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

def build_approx_sc_learnig_data(data_train_path, data_test_path, dict_train_path, dict_test_path, outpath): 

    #training data
    Wd, patches = load_dataset_and_dict(datapath=data_train_path, dictpath=dict_train_path)
    data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
    np.save(outpath + 'trainset.npy', data_lable_dict)

    #test data
    if not data_test_path == '':
        Wd, patches = load_dataset_and_dict(datapath=data_test_path, dictpath=dict_test_path)
        data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
        np.save(outpath + '\testset.npy', data_lable_dict)


def trainset_gen(data_train_path, valid_ratio=0.2):
    dt = np.load(data_train_path)
    data_dict = dt.item()

    train_offset = int(valid_ratio*len(data_dict['X']))

    input   = data_dict['X']
    labels  = data_dict['Y'] 

    input  -= np.mean(input, axis=0)
    input  /= np.std(input,  axis=0)
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


if __name__ == '__main__':
    build_approx_sc_learnig_data(r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\patches_for_traindict\train_set_3000.npy', \
                                 '', r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\learned_dict\Wd_iter100000_alpha0.5_stdthrsh6.npy', \
                                   '', r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\lcod_trainset')