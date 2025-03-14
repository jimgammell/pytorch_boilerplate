import os
from enum import Enum
from typing import Optional, Callable, Union
from tqdm import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common import *
from utils.aes import AES_SBOX, gf256_prod
from ..base_dataset import BaseDataset

G_PERM = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02], dtype=np.uint8)

DATABASE_FILENAMES = [
    r'ascadv2-stm32-conso-raw-traces1.h5',
    r'ascadv2-stm32-conso-raw-traces2.h5',
    r'ascadv2-stm32-conso-raw-traces3.h5',
    r'ascadv2-stm32-conso-raw-traces4.h5',
    r'ascadv2-stm32-conso-raw-traces5.h5',
    r'ascadv2-stm32-conso-raw-traces6.h5',
    r'ascadv2-stm32-conso-raw-traces7.h5',
    r'ascadv2-stm32-conso-raw-traces8.h5',
]
STATS_CACHE_FILENAME = r'ascadv2_stats_cache.npy'

def create_binary_trace_dataset(root: str, chunk_size=100):
    out_path = os.path.join(root, 'ascadv2_traces.npy')
    if os.path.exists(out_path):
        return
    assert all(os.path.exists(os.path.join(root, x)) for x in DATABASE_FILENAMES)
    binary_database = np.memmap(out_path, dtype=np.int8, mode='w+', shape=(int(800e3), int(1e6)))
    idx = 0
    progress_bar = tqdm(total=int(800e3))
    for database_filename in DATABASE_FILENAMES:
        with h5py.File(os.path.join(root, database_filename), 'r') as f:
            trace_database = f['traces']
            for chunk_idx in range(trace_database.shape[0]//chunk_size):
                traces = trace_database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]
                binary_database[idx:idx+chunk_size, :] = traces
                idx += chunk_size
                progress_bar.update(chunk_size)
    binary_database.flush()
    del binary_database

class ASCADv2_Targets(Enum):
    UNPROTECTED = 'unprotected' # masking and permutation disabled
    MASKED = 'masked' # permutation disabled
    PERMUTED = 'permuted' # masking disabled
    FULL = 'full' # fully-protected ASCADv2 implementation

class ASCADv2(BaseDataset):
    @property
    def shape(self):
        return (1, 1000000)
    def _get_mean_and_std(self, chunk_size=100):
        while os.path.exists(os.path.join(self.root, 'dontcomputestats')):
            pass
        if not os.path.exists(os.path.join(self.root, STATS_CACHE_FILENAME)):
            with open(os.path.join(self.root, 'dontcomputestats'), 'w') as _: pass
            try:
                logger.info('Computing and caching ASCADv2 statistics...')
                _ = self.load_datapoint(0)
                assert self.databases is not None
                mean = np.zeros(self.shape, dtype=np.float32)
                std = np.zeros(self.shape, dtype=np.float32)
                count = 0
                progress_bar = tqdm(total=2*len(self)//chunk_size)
                for database in self.databases:
                    assert isinstance(trace_database := database['traces'], h5py.Dataset)
                    for chunk_idx in range(trace_database.shape[0]//chunk_size):
                        traces = np.array(trace_database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :], dtype=np.float32)
                        mean = (count/(count+chunk_size))*mean + (chunk_size/(count+chunk_size))*traces.mean(axis=0).reshape(1, -1)
                        count += chunk_size
                        progress_bar.update(1)
                for database in self.databases:
                    assert isinstance(trace_database := database['traces'], h5py.Dataset)
                    for chunk_idx in range(trace_database.shape[0]//chunk_size):
                        traces = np.array(trace_database[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :], dtype=np.float32)
                        std = (count/(count+chunk_size))*std + (chunk_size/(count+chunk_size))*((traces - mean)**2).mean(axis=0).reshape(1, -1)
                        progress_bar.update(1)
                std = np.sqrt(std)
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
        #mean, _ = self._get_mean_and_std()
        #return mean
        return None
    @property
    def std(self):
        #_, std = self._get_mean_and_std()
        #return std
        return None

    def __init__(self, root: str, target: Union[str, ASCADv2_Targets], transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        create_binary_trace_dataset(self.root)
        self.target = target if isinstance(target, ASCADv2_Targets) else ASCADv2_Targets(target)
        self.databases = None
        #self._get_mean_and_std()
        self.transform = transform or transforms.Compose([
            #transforms.Lambda(lambda x: (x - self.mean)/(self.std + 1e-6)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float))
        ])
        self.target_transform = target_transform or transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))

    def compute_target(self, metadata):
        key = metadata['key']
        plaintext = metadata['plaintext']
        masks = metadata['masks']
        ciphertext = metadata['ciphertext']
        perm_params = masks[:4] & 0x0F
        perm_indices = G_PERM[G_PERM[G_PERM[G_PERM[(15-np.arange(16))^perm_params[0]]^perm_params[1]]^perm_params[2]]^perm_params[3]]
        alpha = masks[18]
        beta = masks[17]
        if self.target in [ASCADv2_Targets.UNPROTECTED, ASCADv2_Targets.MASKED]: # targeting permuted SubBytes effectively disables permutation countermeasure
            out = np.uint8(AES_SBOX[plaintext[perm_indices] ^ key[perm_indices]])
        else:
            out = np.uint8(AES_SBOX[plaintext ^ key])
        if self.target in [ASCADv2_Targets.PERMUTED, ASCADv2_Targets.UNPROTECTED]: # targeting masked SubBytes effectively disables masking
            out = gf256_prod(np.full_like(out, alpha), out) ^ beta
        return out

    def load_datapoint(self, idx):
        if self.databases is None: # init lazily to avoid multiprocessing-related issues
            self.databases = [
                h5py.File(os.path.join(self.root, filename), 'r') for filename in DATABASE_FILENAMES
            ]
            self.database_lengths = [x['traces'].shape[0] for x in self.databases]
            self.cumulative_database_lengths = np.cumsum([0] + self.database_lengths)
        database_idx = np.logical_and(idx >= self.cumulative_database_lengths[:-1], idx < self.cumulative_database_lengths[1:]).argmax()
        point_idx = idx - self.cumulative_database_lengths[database_idx]
        trace = np.array(self.databases[database_idx]['traces'][point_idx], dtype=np.int8).astype(np.float32).reshape(*self.shape)
        metadata = self.databases[database_idx]['metadata'][point_idx]
        metadata = {
            'key': np.array(metadata['key'], dtype=np.uint8),
            'plaintext': np.array(metadata['plaintext'], dtype=np.uint8),
            'ciphertext': np.array(metadata['ciphertext'], dtype=np.uint8),
            'masks': np.array(metadata['masks'], dtype=np.uint8)
        }
        return trace, metadata
    
    def __getitem__(self, idx):
        trace, metadata = self.load_datapoint(idx)
        target = self.compute_target(metadata)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target
    
    def __len__(self):
        if not hasattr(self, 'cumulative_database_lengths'):
            _ = self.load_datapoint(0)
        return self.cumulative_database_lengths[-1]