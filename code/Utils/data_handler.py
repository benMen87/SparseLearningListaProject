"""
Module for handling data - load, preprocess and serve for train.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from data_loader import DataLoader


class DataHandlerNoise(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, noise_sigma, norm_val=255, ds_name):
        super(DataHandlerNoise, self).__init__(valid_ratio)
        self.sigma = float(noise_sigma) / norm_val
        self.load_data(name=ds_name, norm_val=norm_val)

    def xy_gen(self, target, batch_size):
        """Add noise to targer im yeild batches"""
        data_in = target + np.random.normal(0, self.sigma, target.shape)
        X_batch = self.Batch(batch_size, data_in)
        Y_batch = self.Batch(batch_size, target)

        for X, Y in zip(X_batch, Y_batch):
            yield X, Y

    def train_gen(self, batch_size):
        return self.xy_gen(self.train, batch_size)

    def valid_gen(self, batch_size):
        return self.xy_gen(self.valid, batch_size)

    def valid_gen(self, batch_size):
        return self.xy_gen(self.valid, batch_size)

        

class DataHandlerBase(DataLoader):
    """
    Base class for:
    Load + Preprocess + handle batches
    """
    __metaclass__ = ABCMeta

    class DataNotLoadedException(Exception):
        pass


    def __init__(self, valid_ratio=0.2):
        self.train = None
        self.test = None
        self.valid = None
        self.valid_ratio = 0.2

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

    class Batch():

        def __init__(self, batch_size, data):
            self.batch_size = batch_size
            self.batch_num = 0
            self.epoch = 0
            self.data = data
            self.data_len = len(data)

        def __iter__(self):
            return self

        def next(self, run_once=False):
    
            if self.data_len == 0:
                raise  DataNotLoadedException('No data')

            batchs_per_epoch = np.ceil(float(self.data_len) / batch_size)
            if self.batch_num < batchs_per_epoch:
                b = self.data[self.start():self.end()]
            else:
                self.epoch += 1
                self.batch_num = 0
                if run_once:
                    raise StopIteration()
                else:
                    b = self.data[self.start():self.end()]
            self.batch_num += 1

            return b   

        def start(self):
            return self.batch_size * self.batch_num
            
        def end(self):
            np.min(self.batch_size*(1+self.batch_num), self.data_len)            
