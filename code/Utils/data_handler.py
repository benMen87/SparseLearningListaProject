"""
Module for handling data - load, preprocess and serve for train.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from data_loader import DataLoader

class DataNotLoadedException(Exception):
    pass


class DataHandlerBase(DataLoader):
    """
    Base class for:
    Load + Preprocess + handle batches
    """
    __metaclass__ = ABCMeta

    valid_ratio = 0.2 

    def __init__(self, valid_ratio=0.2):
        self.train = None
        self.test = None
        self.valid = None
        
    @staticmethod
    def factory(**kwargs):

        DS_ARGS = {
            'valid_ratio':DataHandlerBase.valid_ratio,
            'ds_name':kwargs['dataset'],
            'norm_val':kwargs['norm_val']
        }

        if kwargs['task'] == 'denoise':
            DS_ARGS['noise_sigma'] = kwargs['noise_sigma']
            return DataHandlerNoise(**DS_ARGS)
        elif kwargs['task'] == 'multi_denoise':
            DS_ARGS['noise_sigma'] = kwargs['noise_sigma']
            DS_ARGS['dup_count'] = kwargs['dup_count']
            return DataHandlerMultipleNoise(**DS_ARGS)
        elif kwargs['task'] == 'inpaint':
            DS_ARGS['inpaint_keep_prob'] = kwargs['inpaint_keep_prob']
            return DataHandlerInpaint(**DS_ARGS)
        else:
            raise BadDsNameOrPath()

    @abstractmethod
    def train_gen(self, batch_size):
        pass

    @abstractmethod
    def valid_gen(self, batch_size):
        pass
    
    @abstractmethod
    def test_gen(self, batch_size):
        pass

    def load_data(self, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            self.train, self.test = self.load_data_for_train(name)
        elif 'path' in kwargs:
            path  = kwargs['path']
            self.train, self.test = self.load_data_for_train(path)
        else:
            raise BadDsNameOrPath('Missing name or path arg')
        self.train /= kwargs.get('norm_val', 1)
        self.test /= kwargs.get('norm_val', 1)

        np.random.shuffle(self.train)
        valid_size = np.int(len(self.train) * self.valid_ratio)
        self.valid = self.train[:valid_size] 
        self.train = self.train[valid_size:]

    @abstractmethod
    def preprocess_data(self, **kwargs):
       pass

    @property
    def shape(self):
        if self.train is None:
            return 0
        else:
            return self.train.shape[1:]


    class Batch():
        def __init__(self, batch_size, data, target=None, run_once=False, use_mask=False):
            self.batch_size = batch_size
            self.batch_num = 0
            self.epoch = 0
            self.data = data
            self.target = target
            self._use_mask = use_mask
            self.curr_batch = (None, None)
            self.data_len = len(data)
            self.batchs_per_epoch = np.ceil(float(self.data_len) / self.batch_size) 
            self.run_once = run_once

        def __iter__(self):
            return self

        def next(self):

            if self.data_len == 0:
                raise  DataNotLoadedException('No data')
            s, e = self.start_end()
            if self.batch_num < self.batchs_per_epoch:
                self.curr_batch = (self.data[s:e], self.target[s:e])
            else:
                self.rewind()
                if self.run_once:
                    raise StopIteration()
                else:
                    self.curr_batch = (self.data[s:e], self.target[s:e])
            self.batch_num += 1
            return self.curr_batch   
        
        def rewind(self):
            self.epoch += 1
            self.batch_num = 0
        
        def mask(self):
            if self._use_mask:
                return (self.curr_batch[0] == self.curr_batch[1]).astype(float)
            else:
                return 1

        def start_end(self):
            start = self.batch_size * self.batch_num
            end = np.minimum(self.batch_size + start, self.data_len)
            if not start < end and not self.run_once:
                self.rewind()
                return self.start_end()
            return start, end

class DataHandlerNoise(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, noise_sigma, ds_name, norm_val=255):
        super(DataHandlerNoise, self).__init__(valid_ratio)
        self.sigma = float(noise_sigma) / norm_val
        self.load_data(name=ds_name, norm_val=norm_val)

    def preprocess_data(self, **kwargs):
        data = kwargs['data']
        data_n = data + np.random.normal(0, self.sigma, data.shape)
        return data_n

    def xy_gen(self, target, batch_size, run_once):
        print(batch_size)
        """Add noise to targer im yeild batches"""
        data_in =  self.preprocess_data(data=target)
        batch = self.Batch(batch_size, data_in, target, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

class DataHandlerMultipleNoise(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, noise_sigma,ds_name, dup_count, norm_val=255):
        super(DataHandlerMultipleNoise, self).__init__(valid_ratio)
        self.sigma = float(noise_sigma) / norm_val
        self.load_data(name=ds_name, norm_val=norm_val)
        self.dup_count = dup_count

    def preprocess_data(self, **kwargs):
        data = kwargs['data']
        data = np.repeat(data, self.dup_count, axis=0)
        data_n = data + np.random.normal(0, self.sigma, data.shape)
        return data_n, data

    def xy_gen(self, target, batch_size, run_once):
        """Add noise to targer im yeild batches"""
        data_in, target =  self.preprocess_data(data=target)
        batch = self.Batch(batch_size, data_in, target, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
       return self.xy_gen(self.valid, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

class DataHandlerInpaint(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, keep_prob, ds_name, norm_val=255):
        super(DataHandlerInpaint, self).__init__(valid_ratio)
        self.drop_prob = 1 - keep_prob
        self.load_data(name=ds_name, norm_val=norm_val)

    def preprocess_data(self, **kwargs):
        data = kwargs['data']
        p = self.drop_prob
        mask = np.random.choice([0,1], size=data.shape, p=[p, 1-p])
        data_n = data * mask
        return data_n, mask

    def xy_gen(self, target, batch_size, run_once):
        """Add noise to target im yeild batches"""
        data_in, mask = self.preprocess_data(data=target)
        batch = self.Batch(batch_size, data_in, target, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)
