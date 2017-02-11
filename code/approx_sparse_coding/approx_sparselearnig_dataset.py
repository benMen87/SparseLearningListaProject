import numpy as np

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
        data_set = np.load(dict_path)
        return data_set
    except:
        raise FileNotFoundError('Cannot find dictionary in specified path %s'%dataset_path)

def compute_patch_sc_pydict(patch_set, Wd):
    sparse_rep = []
    sparse_code = CoD(Wd, alpha=0.5)
    for d in data_set:
        Z = 

def build_approx_sc_learnig_data(data_train_path, data_test_path, dict_train_path, dict_test_path): 

    Wd      = load_dictioary(dict_train_path)
    patches = load_data_set(data_train_path)

