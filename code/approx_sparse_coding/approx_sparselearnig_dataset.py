import numpy as np
import os
import sys

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
        Z = sparse_code.run_cod(X=patch)
        sparse_rep.append(Z)
    sparse_rep = np.asarray(sparse_rep)
    return {'X':patch_set, 'Y':sparse_rep}

def build_approx_sc_learnig_data(data_train_path, data_test_path, dict_train_path, dict_test_path, outpath): 

    #training data
    Wd, patches = load_dataset_and_dict(datapath=data_train_path, dictpath=dict_train_path)
    data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
    np.save(outpath + 'trainset.npy', data_lable_dict)

    #test data
    Wd, patches = load_dataset_and_dict(datapath=data_test_path, dictpath=dict_test_path)
    data_lable_dict = compute_patch_sc_pydict(patch_set=patches, Wd=Wd)
    np.save(outpath + 'testset.npy', data_lable_dict)


build_approx_sc_learnig_data(r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\patches_for_traindict\train_set_3000.npy', '' \
    , r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\learned_dict\Wd_iter50000_alpha0.5_stdthrsh.py.npy', '', './')



