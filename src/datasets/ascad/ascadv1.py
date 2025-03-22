from typing import Union, Callable, Optional, Tuple, List
import os
import time
from enum import Enum
from dataclasses import dataclass

from tqdm import tqdm
import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from common import *
from ..base_dataset import BaseDataset
from utils.aes import *
from ..utils import get_trace_sample_stats

class ASCADv1_Targets(Enum):
    SUBBYTES = 'subbytes'
    R = 'r'
    ROUT = 'r_out'
    SUBBYTES_XOR_R = 'subbytes_xor_r'
    SUBBYTES_XOR_ROUT = 'subbytes_xor_rout'
    KEY = 'key'
    PLAINTEXT = 'plaintext'
ASCADv1_Targets_t = Union[ASCADv1_Targets, str]

@dataclass
class _Database:
    traces: Union[torch.Tensor, h5py.Dataset]
    plaintext: Union[torch.Tensor, NDArray[np.uint8]]
    key: Union[torch.Tensor, NDArray[np.uint8]]
    masks: Union[torch.Tensor, NDArray[np.uint8]]
    store_in_ram: bool

    def post_init_ram(self):
        assert self.store_in_ram
        if isinstance(self.traces, torch.Tensor):
            pass
        else:
            assert isinstance(self.traces, np.ndarray) and self.traces.dtype == np.int8
            self.traces = torch.from_numpy(self.traces)
        if isinstance(self.plaintext, torch.Tensor):
            pass
        else:
            assert isinstance(self.plaintext, np.ndarray) and self.plaintext.dtype == np.uint8
            self.plaintext = torch.from_numpy(self.plaintext)
        if isinstance(self.key, torch.Tensor):
            pass
        else:
            assert isinstance(self.key, np.ndarray) and self.key.dtype == np.uint8
            self.key = torch.from_numpy(self.key)
        if isinstance(self.masks, torch.Tensor):
            pass
        else:
            assert isinstance(self.masks, np.ndarray) and self.masks.dtype == np.uint8
            self.masks = torch.from_numpy(self.masks)
        assert self.traces.dtype == torch.int8
        assert self.plaintext.dtype == torch.uint8
        assert self.key.dtype == torch.uint8
        assert self.masks.dtype == torch.uint8
        self.traces.share_memory_()
        self.plaintext.share_memory_()
        self.key.share_memory_()
        self.masks.share_memory_()
    
    def post_init_disk(self):
        assert not(self.store_in_ram)
        assert isinstance(self.traces, h5py.Dataset) and self.traces.dtype == np.int8
        assert isinstance(self.plaintext, np.ndarray) and self.plaintext.dtype == np.uint8
        assert isinstance(self.key, np.ndarray) and self.key.dtype == np.uint8
        assert isinstance(self.masks, np.ndarray) and self.masks.dtype == np.uint8

    def __post_init__(self):
        if self.store_in_ram:
            self.post_init_ram()
        else:
            self.post_init_disk()
        assert len(self.traces) == len(self.plaintext) == len(self.key) == len(self.masks)
        self.length = len(self.traces)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.store_in_ram:
            return self.traces[idx], self.plaintext[idx], self.key[idx], self.masks[idx]
        else:
            return tuple(map(lambda x: torch.from_numpy(x[idx]), (self.traces, self.plaintext, self.key, self.masks)))
    
    def __len__(self) -> int:
        return self.length

_data: Optional[_Database] = None
def _init_shared_dataset(database_path: str):
    global _data
    if _data is not None:
        return
    with h5py.File(database_path, 'r') as f:
        traces = np.array(f['traces'])
        metadata_f = f['metadata']
        _data = _Database(traces=traces, plaintext=metadata_f['plaintext'], key=metadata_f['key'], masks=metadata_f['masks'], store_in_ram=True)

class _ASCADv1(Dataset):
    DATABASE_SHAPE: Optional[Tuple[int, ...]] = None
    TRACE_SHAPE: Optional[Tuple[int, ...]] = None
    PROFILE_INDICES: Optional[NDArray] = None
    ATTACK_INDICES: Optional[NDArray[np.int64]] = None
    DATABASE_FILENAME: Optional[str] = None
    STATS_CACHE_FILENAME: Optional[str] = None

    def __init__(self,
        root: str,
        train: bool = True,
        target: Union[ASCADv1_Targets_t, List[ASCADv1_Targets_t]] = ASCADv1_Targets.SUBBYTES,
        store_in_ram: bool = False
    ):
        super().__init__()

        assert self.DATABASE_SHAPE is not None
        assert self.TRACE_SHAPE is not None
        assert self.PROFILE_INDICES is not None
        assert self.ATTACK_INDICES is not None
        assert self.DATABASE_FILENAME is not None
        assert self.STATS_CACHE_FILENAME is not None

        self.root = root
        self.train = train
        self.target = target if isinstance(target, list) else [target]
        self.target = [ASCADv1_Targets(x) if isinstance(x, str) else x for x in self.target]
        self.store_in_ram = store_in_ram
        self.database_path = os.path.join(self.root, self.DATABASE_FILENAME)
        self.stats_cache_path = os.path.join(self.root, self.STATS_CACHE_FILENAME)
        self.needs_to_setup = True
        self.indices = self.PROFILE_INDICES if self.train else self.ATTACK_INDICES
        if self.store_in_ram:
            _init_shared_dataset(self.database_path)
    
    def setup(self):
        global _data
        if self.store_in_ram:
            while _data is None:
                time.sleep(0.1)
            self.data = _data
        else:
            self.data_file = h5py.File(self.database_path, 'r')
            self.data = _Database(
                self.data_file['traces'], self.data_file['metadata']['plaintext'], self.data_file['metadata']['key'], self.data_file['metadata']['masks'], store_in_ram=False # type: ignore
            )
        self.aes_sbox = torch.from_numpy(AES_SBOX)
        self.mean, self.std = get_trace_sample_stats(self.data.traces, self.stats_cache_path, indices=self.PROFILE_INDICES)
        assert self.TRACE_SHAPE is not None
        self.mean = torch.from_numpy(self.mean).reshape(*self.TRACE_SHAPE).to(torch.float)
        self.std = torch.from_numpy(self.std).reshape(*self.TRACE_SHAPE).to(torch.float).clamp_(min=1e-6)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float).reshape(*self.TRACE_SHAPE)),
            transforms.Lambda(lambda x: (x - self.mean)/self.std)
        ])
        self.target_transform = transforms.Lambda(lambda x: x.to(torch.long))
        self.needs_to_setup = False

    def compute_target(self, key: torch.Tensor, plaintext: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        r = torch.cat([torch.zeros((2,), dtype=torch.uint8, device=masks.device), masks[:-2]])
        r_in = masks[-2]
        r_out = masks[-1]
        out = []
        if ASCADv1_Targets.SUBBYTES in self.target:
            out.append(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)])
        if ASCADv1_Targets.R in self.target:
            out.append(r)
        if ASCADv1_Targets.ROUT in self.target:
            out.append(r_out.unsqueeze(0))
        if ASCADv1_Targets.SUBBYTES_XOR_R in self.target:
            out.append(torch.bitwise_xor(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)], r.unsqueeze(0)))
        if ASCADv1_Targets.SUBBYTES_XOR_ROUT in self.target:
            out.append(torch.bitwise_xor(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)], r_out.unsqueeze(0)))
        if ASCADv1_Targets.KEY in self.target:
            out.append(key)
        if ASCADv1_Targets.PLAINTEXT in self.target:
            out.append(plaintext)
        assert len(out) > 0
        out = torch.cat(out)
        return out
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[idx]
        if self.needs_to_setup:
            self.setup()
        trace, plaintext, key, masks = self.data[idx]
        target = self.compute_target(key, plaintext, masks)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        assert isinstance(trace, torch.Tensor) and isinstance(target, torch.Tensor)
        return trace, target
    
    def __len__(self) -> int:
        return len(self.indices)

    def __del__(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

class ASCADv1_Fixed(_ASCADv1):
    DATABASE_SHAPE = (60000, 100000)
    TRACE_SHAPE = (1, 100000)
    PROFILE_INDICES = np.arange(0, 50000, dtype=np.int64)
    ATTACK_INDICES = np.arange(50000, 60000, dtype=np.int64)
    DATABASE_FILENAME = r'ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5'
    STATS_CACHE_FILENAME = r'ascadv1_fixed_stats_cache.npy'

class ASCADv1_Var(_ASCADv1):
    DATABASE_SHAPE = (300000, 250000)
    TRACE_SHAPE = (1, 250000)
    PROFILE_INDICES = np.concatenate([np.arange(0, 300000, 3, dtype=np.int64), np.arange(1, 300000, 3, dtype=np.int64)])
    ATTACK_INDICES = np.arange(2, 300000, 3, dtype=np.int64)
    DATABASE_FILENAME = r'atmega8515-raw-traces.h5'
    STATS_CACHE_FILENAME = r'ascadv1_var_stats_cache.npy'