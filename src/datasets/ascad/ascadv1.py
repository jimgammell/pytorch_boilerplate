from typing import Union, Callable, Optional
import os
from enum import Enum

from tqdm import tqdm
import h5py
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from common import *
from ..base_dataset import BaseDataset
from utils.aes import *

DATABASE_FILENAME = r'atmega8515-raw-traces.h5'
STATS_CACHE_FILENAME = r'ascadv1_stats_cache.npy'

def create_binary_trace_dataset(root: str, chunk_size: int = 100):
    out_path = os.path.join(root, 'ascadv1_var_traces.npy')
    if os.path.exists(out_path):
        return
    assert os.path.exists(os.path.join(root, DATABASE_FILENAME))
    binary_database = np.memmap(out_path, dtype=np.int8, mode='w+', shape=(300000, 250000))
    idx = 0
    progress_bar = tqdm(total=300000)
    with h5py.File(os.path.join(root, DATABASE_FILENAME), 'r') as f:
        trace_database = f['traces']
        for chunk_idx in range(300000//chunk_size):
            traces = trace_database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]
            binary_database[idx:idx+chunk_size, :] = traces
            idx += chunk_size
            progress_bar.update(chunk_size)
    binary_database.flush()
    del binary_database

class ASCADv1_Targets(Enum):
    FULL = 'full'
    UNPROTECTED = 'unprotected'

class ASCADv1(BaseDataset):
    database_shape = (300000, 250000)
    @property
    def shape(self):
        return (1, 250000)
    def _get_mean_and_std(self, chunk_size=100):
        while os.path.exists(os.path.join(self.root, 'dontcomputestats')):
            pass
        if not os.path.exists(os.path.join(self.root, STATS_CACHE_FILENAME)):
            with open(os.path.join(self.root, 'dontcomputestats'), 'w') as _: pass
            try:
                logger.info('Computing and caching ASCADv1 statistics...')
                _ = self.load_datapoint(0)
                assert self.database is not None
                mean = np.zeros(self.shape, dtype=np.float32)
                var = np.zeros(self.shape, dtype=np.float32)
                count = 0
                progress_bar = tqdm(total=2*len(self))
                for chunk_idx in range(len(self)//chunk_size):
                    traces = np.array(self.database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :], dtype=np.float32)
                    mean = (count/(count+chunk_size))*mean + (chunk_size/(count+chunk_size))*traces.mean(axis=0).reshape(1, -1)
                    count += chunk_size
                    progress_bar.update(chunk_size)
                for chunk_idx in range(len(self)//chunk_size):
                    traces = np.array(self.database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :], dtype=np.float32)
                    var = (count/(count+chunk_size))*var + (chunk_size/(count+chunk_size))*((traces - mean)**2).mean(axis=0).reshape(1, -1)
                    progress_bar.update(chunk_size)
                std = np.sqrt(var)
                np.save(os.path.join(self.root, STATS_CACHE_FILENAME), np.stack([mean, std]))
                self.databases = None
            finally:
                os.remove(os.path.join(self.root, 'dontcomputestats'))
        mean_and_std = np.load(os.path.join(self.root, STATS_CACHE_FILENAME))
        mean = mean_and_std[0, ...]
        std = mean_and_std[1, ...]
        return mean, std
    @property
    def mean(self):
        mean, _ = self._get_mean_and_std()
        return mean
    @property
    def std(self):
        _, std = self._get_mean_and_std()
        return std
    
    def __init__(self, root: str, train: bool = True, target: Union[str, ASCADv1_Targets] = 'full', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, store_in_ram: bool = False):
        super().__init__()
        self.root = root
        self.train = train
        self.target = target if isinstance(target, ASCADv1_Targets) else ASCADv1_Targets(target)
        self.transform = transform or transforms.Compose([transforms.Lambda(lambda x: (x - self.mean)/(self.std+1e-6)), transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float))])
        self.target_transform = target_transform or transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.store_in_ram = store_in_ram
        create_binary_trace_dataset(self.root)
        self.database_path = os.path.join(self.root, 'ascadv1_var_traces.npy')
        assert os.path.exists(self.database_path)
        if store_in_ram:
            self.database = np.array(np.memmap(self.database_path, mode='r', shape=self.database_shape))
        else:
            self.database = None
        self.metadata = None
    
    def compute_target(self, metadata):
        key = metadata['key']
        plaintext = metadata['plaintext']
        masks = metadata['masks']
        r = np.concatenate([np.zeros(2, dtype=np.uint8), masks[:-2]])
        r_in = masks[-2]
        r_out = masks[-1]
        if self.target == ASCADv1_Targets.UNPROTECTED:
            out = np.uint8(AES_SBOX[plaintext ^ key] ^ r_out)
        elif self.target == ASCADv1_Targets.FULL:
            out = np.uint8(AES_SBOX[plaintext ^ key])
        else:
            assert False
        return out
    
    def load_datapoint(self, idx):
        if self.database is None:
            self.full_database = h5py.File(os.path.join(self.root, DATABASE_FILENAME), 'r')#
            self.database = self.full_database['traces'] #np.memmap(self.database_path, mode='r', shape=self.database_shape)
        if self.metadata is None:
            #with h5py.File(os.path.join(self.root, DATABASE_FILENAME), mode='r', swmr=True) as f:
            metadatabase = self.full_database['metadata']#f['metadata']
            self.metadata = {
                'key': np.array(metadatabase['key'], dtype=np.uint8),
                'plaintext': np.array(metadatabase['plaintext'], dtype=np.uint8),
                'masks': np.array(metadatabase['masks'], dtype=np.uint8)
            }
        trace = np.array(self.database[idx, :], dtype=np.float32).reshape(*self.shape)
        metadata = {key: val[idx] for key, val in self.metadata.items()}
        return trace, metadata
    
    def __getitem__(self, idx: int):
        trace, metadata = self.load_datapoint(idx)
        target = self.compute_target(metadata)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target
    
    def __len__(self):
        return self.database_shape[0]