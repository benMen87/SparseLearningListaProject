import sys 
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from Utils import db_tools


if __name__ == '__main__':

    TRIANING_PATCHS_PATH = r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\patches_for_traindict\maxiter_150000_size_30000_thrsh_1_train_set.npy'
    TEST_PATCHS_PATH = ''
    DICTIONARY_PATH =  r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\learned_dict\Wd_maxiter_150000_size_30000_thrsh_1_train_set.npy'
    OUTPUT_PATH = r'C:\Users\Hillel\Desktop\Msc3\sparse\final_project\lcod_trainset'

    db_tools.build_approx_sc_learnig_data(TRIANING_PATCHS_PATH, TEST_PATCHS_PATH, DICTIONARY_PATH,
                                          DICTIONARY_PATH, OUTPUT_PATH)