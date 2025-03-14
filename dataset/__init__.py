
from typing import Dict, List, Tuple, Union
import threading, time
import numpy as np
from queue import Queue 
from PIL.Image import Image as PILImage
from abc import ABC, abstractmethod
 

################################################################################
# A Parallel Dataset class: Preprocessing and generating data batches through  #
# sub processes.                                                               #
################################################################################
class ParallelDataset():
    def __init__(self, sample_count:int, get_data_by_ids_func,
        batch_size:Union[int, List[int]] = 256, shuffle = True, 
        buffer_size = 64, drop_last = False, random_seed = None, 
        return_samp_n = True) -> None:
        '''
        A basic dataset class, whose subclasses should realize 
        `__get_data_by_ids__(self, ids)` which is used for obtaining data by ids.
        `sample_count`: Total count of data samples
        `get_data_by_ids_func`: the function that obtain data by id.
        `batch_size`: The batch size of data sample to be generated. If a list or 
            one-dimensional array is passed in, randomly select numbers from 
            the list each time as a batch size.
        `shuffle`: Whether shuffle the data to be generated.
        `buffer_size`: The buffer size of queue.
        `drop_last`: Whether discard the last batch of data. If not, combine them
            into the first batch of data in the next epoch.
        '''
        self.sample_count = sample_count
        self.set_batch_size(batch_size)
        # batch_size = [batch_size] if type(batch_size) == int else batch_size
        # self.batch_size = np.array([min(bs, sample_count) for bs in batch_size])
        self.shuffle = shuffle
        self.return_samp_n = return_samp_n
        self.rng = np.random.default_rng(random_seed)
        self.select_ids = np.array(range(sample_count))
        if shuffle: 
            self.rng.shuffle(self.select_ids)
        self.drop_last = drop_last
        self.now_buffer_i = 0 # the idex of data has added into buffer
        self.now_yield_i = 0 # the idex of data has yielded
        self.buffer_size = buffer_size
        self.buffer = Queue()
        self.is_loading_data = False
        self.__get_data_by_ids__ = get_data_by_ids_func
        self.__fill_buffer__()

    def set_batch_size(self, batch_size:Union[int, List[int]]):
        if type(batch_size) != list and type(batch_size) != int: raise
        if type(batch_size) == list and len(batch_size) == 0: raise
        if type(batch_size) == int and batch_size <= 0: raise
        batch_size = [batch_size] if type(batch_size) != list else batch_size
        self.batch_size = np.array([min(bs, self.sample_count) for bs in batch_size])

    def __get_data_by_ids__(self, ids):
        raise

    def __fill_buffer__(self):
        if self.is_loading_data:
            return
        self.is_loading_data = True 
        def fill_buffer(): 
            while self.buffer.qsize() < self.buffer_size:
                bs = self.rng.choice(self.batch_size)
                tail_i = self.now_buffer_i + bs
                ids = self.select_ids[self.now_buffer_i:tail_i]
                if tail_i >= self.sample_count:
                    self.select_ids = np.array(range(self.sample_count))
                    if self.shuffle:
                        self.rng.shuffle(self.select_ids)
                    if tail_i > self.sample_count and self.drop_last:
                        self.now_buffer_i = 0
                        continue
                    self.now_buffer_i = tail_i - self.sample_count
                    extra_ids = self.select_ids[:self.now_buffer_i]
                    ids = np.concatenate([ids, extra_ids], 0)
                else:
                    self.now_buffer_i = tail_i
                d = self.__get_data_by_ids__(ids)
                self.buffer.put((d, len(ids)))
            self.is_loading_data = False  
        threading.Thread(target = fill_buffer).start() 
    
    def __len__(self): 
        if len(self.batch_size) > 1:
            print('The number of data batches is not accurate since `batch_size` is a list')
        bs = self.batch_size.mean()
        if self.drop_last:
            return int(np.floor(self.sample_count/bs))
        return int(np.ceil(self.sample_count/bs))

    def __iter__(self): 
        self.now_yield_i = 0
        return self

    def __next__(self):
        if self.now_yield_i >= self.sample_count:
            raise StopIteration
        if self.buffer.qsize() <= self.buffer_size/2:
            self.__fill_buffer__() 
        t = 0  
        while self.buffer.qsize() == 0:  
            print('\r', "Waiting data: %d s"%t, end='')
            time.sleep(1) 
            t += 1  
        d, data_n = self.buffer.get()
        self.now_yield_i += data_n
        if self.return_samp_n:
            return d, data_n
        else:
            return d



class BaseEditData(ABC):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    @abstractmethod
    def dataset_name(self):
        '''return dataset name'''
        raise
    