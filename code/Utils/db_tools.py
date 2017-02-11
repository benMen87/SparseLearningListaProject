from sklearn.feature_extraction import image
import tarfile as tar
import numpy as np
from PIL import Image
import os

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
                print('Error when loading trainset will try to rebuild train set') 
        # else build it
        train_data = load_train_data_to_mem(db_fullpath, patch_size, std_thrsh, train_size)
        train_data = np.append(train_data, saved_train_data, axis=0)
        if savefile:
            np.save(train_fullpath, train_data)
        
        return train_data
