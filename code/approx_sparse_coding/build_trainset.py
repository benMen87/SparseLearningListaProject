import sys 
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/..')
from Utils import db_tools


if __name__ == '__main__':

    TRIANING_PATCHS_PATH = dir_path  + '/../../patches_for_traindict/train.npy'
    TEST_PATCHS_PATH = dir_path  + '/../../patches_for_traindict/test.npy'
    DICTIONARY_PATH =  dir_path  + '/../../convdict2d_data/Wd.npy'
    OUTPUT_PATH = dir_path  + '/../../convdict2d_data/'

    db_tools.build_approx_sc_learnig_data(TRIANING_PATCHS_PATH, TEST_PATCHS_PATH,
                                          DICTIONARY_PATH, DICTIONARY_PATH, OUTPUT_PATH)

