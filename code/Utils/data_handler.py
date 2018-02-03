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
        elif kwargs['task'] == 'doc_clean':
            return DataHandlerSP(**DS_ARGS)
        elif kwargs['task'] == 'multi_denoise':
            DS_ARGS['noise_sigma'] = kwargs['noise_sigma']
            DS_ARGS['dup_count'] = kwargs['dup_count']
            return DataHandlerMultipleNoise(**DS_ARGS)
        elif kwargs['task'] == 'inpaint':
            DS_ARGS['inpaint_keep_prob'] = kwargs['inpaint_keep_prob']
            return DataHandlerInpaint(**DS_ARGS)
        elif kwargs['task'] == 'denoise_dynamicthrsh':
            return DataHandler(**DS_ARGS)
        elif kwargs['task'] == 'deblur':
            return DataHandler(**DS_ARGS)
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

        print('train size: {}, valid size: {}'.format(len(self.train),valid_size))

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
            self._batch_size = batch_size
            self._batch_num = 0
            self._epoch = 0
            self._data = data
            self._target = target if target is not None else data
            self._use_mask = use_mask
            self.curr_batch = (None, None)
            self._data_len = len(data)
            self._batchs_per_epoch = np.ceil(float(self._data_len) / self._batch_size) 
            self._run_once = run_once

        def __iter__(self):
            return self

        def next(self):

            if self._data_len == 0:
                raise  DataNotLoadedException('No data')
            s, e = self._start_end()
            if self._batch_num < self._batchs_per_epoch:
                self._curr_batch = (self._data[s:e], self._target[s:e])
            else:
                self._rewind()
                if self._run_once:
                    raise StopIteration()
                else:
                    self._curr_batch = (self._data[s:e], self._target[s:e])
            self._batch_num += 1
            return self._curr_batch   

        def _rewind(self):
            self._epoch += 1
            self._batch_num = 0

        def mask(self):
            if self._use_mask:
                return (self._curr_batch[0] == self._curr_batch[1]).astype(float)
            else:
                return 1

        def _start_end(self):
            start = self._batch_size * self._batch_num
            end = np.minimum(self._batch_size + start, self._data_len)
            if not start < end and not self._run_once:
                self._rewind()
                return self._start_end()
            return start, end

        @property
        def batch_num(self):
            return self._batch_num

        @property
        def epoch_num(self):
            return self._epoch

        @property
        def size(self):
            return self._data_len

class DataHandler(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, ds_name, norm_val=255):
        super(DataHandler, self).__init__(valid_ratio)
        self.load_data(name=ds_name, norm_val=norm_val)

    def preprocess_data(self, **kwargs):
        pass

    def xy_gen(self, data_in, batch_size, run_once):
        batch = self.Batch(batch_size, data_in, data_in, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

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
        """Add noise to target in yeild batches"""
        data_in =  self.preprocess_data(data=target)
        batch = self.Batch(batch_size, data_in, target, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

class DataHandlerNoiseDup(DataHandlerBase):
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
        batch = self.Batch(batch_size, data_in, target, run_once, use_mask=True)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

class DataHandlerSP(DataHandlerBase):
    """
    Data handler for train session with noise.
    """
    def __init__(self, valid_ratio, ds_name, norm_val=255):
        super(DataHandlerSP, self).__init__(valid_ratio)
        self.load_data(name=ds_name, norm_val=norm_val)

    def preprocess_data(self, **kwargs):
        
        # TODO: add as args to class
        s_vs_p = 0.5
        amount = 0.05

        data = kwargs['data']
        data_n = np.empty(shape=data.shape)

        for im_id, im in zip(range(data.shape[0]), data):
            
            im = np.squeeze(im)
            # Salt mode
            num_salt = np.ceil(amount * im.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                        for i in im.shape]
            im[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* im.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in im.shape]
            im[coords] = 0
            data_n[im_id,...] = im[...,None]
        return data_n

    def xy_gen(self, target, batch_size, run_once):
        """Add noise to target in yeild batches"""
        data_in =  self.preprocess_data(data=target.copy())
        batch = self.Batch(batch_size, data_in, target, run_once)
        return batch

    def train_gen(self, batch_size, run_once=False):
        return self.xy_gen(self.train, batch_size, run_once)

    def valid_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.valid, batch_size, run_once)

    def test_gen(self, batch_size, run_once=True):
        return self.xy_gen(self.test, batch_size, run_once)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dh = DataHandlerSP(0.1, 'docs')
    dt_n =  dh.preprocess_data(data=dh.train[:5,...])
    plt.imshow(dt_n[0,...,0], cmap='gray')
    plt.show()