"""

"""
import os
import numpy as np
import stl10_input


DEFAULT_DATA_PATH = '/data/hillel/data_sets/'

class LoadNpzData(object):

    def __init__(self):
        pass

    def load(self, path, tr_key='TRAIN', vl_key='VAL', ts_key='TEST'):
        data = np.load(path)
        if  len(data[vl_key]):
           train = np.concatenate((data[tr_key], data[vl_key]), axis=0)
        else:
            train = data[tr_key]
        return train, data[ts_key]

class Pascal(LoadNpzData):
    """Load pascal dataset saved as npz"""    
    def __init__(self):
        pass
    
    def load(self, large=True):
        path = DEFAULT_DATA_PATH + large*'pascal320.npz' + (1-large)*'pascal120.npz'
        train, test = super(Pascal, self).load(path)
        return train, test

class Berkeley(LoadNpzData):
    """Load pascal dataset saved as npz"""    
    def __init__(self):
        pass
    
    def load(self, gray=True):
        path = DEFAULT_DATA_PATH + 'berkeley_db_321X321.npz'
        train, test = super(Berkeley, self).load(path)
        return train, test

class Stl10(object):
    """Load STL10 saved as binary file"""
    def __init__(self):
        pass

    def load(gray=True):
       train, test = stl10_input.load_data(grayscale=gray)
       return train, test

class LoadDataFiles(object):
    """Load image files from test dir and train dir"""
    def __init__(self):
        pass

    def load(dir, open_files=False):
        
        data = [f for f in listdir(dir) if isfile(join(dir, f))]

        if open_files:
            raise NotImplementedError('loading files is not implemented')
        return data

    

class DataLoader(Pascal, Stl10, LoadDataFiles):
    
    class BadDsNameOrPath(Exception):
        pass

    dataset_loaders = {
            'pascal': Pascal().load,
            'stl10': Stl10().load,
            'berkeley': Berkeley().load,
        }

    def __init__(self):
        pass

    def load_data_for_train(self, ds_name_or_path):
        train = None
        test = None

        if ds_name_or_path in self.dataset_loaders.keys():
            ds_name = ds_name_or_path
            train, test = self.dataset_loaders[ds_name]()
        elif os.path.isdir(ds_name_or_path):
            path = ds_name_or_path
            train = LoadDataFiles.load(path)
        elif  isinstance(ds_name_or_path, list) or isinstance(ds_name_or_path, tuple):
            if os.path.isdir(ds_name_or_path[0]) and os.path.isdir(ds_name_or_path[1]):
                train = LoadDataFiles.load(ds_name_or_path[0])
                test = LoadDataFiles.load(ds_name_or_path[1])
        if train is None and test is None:
            raise self.BadDsNameOrPath('Bas name or path {}'.format(ds_name_or_path))
        return train.astype('float32'), test.astype('float32')
