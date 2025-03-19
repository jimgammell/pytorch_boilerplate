from enum import Enum
from typing import Union, List, Callable, Optional, Tuple
import os
import ctypes
from dataclasses import dataclass
import multiprocessing

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import get_trace_sample_stats
from utils.aes import AES_SBOX

DATABASE_PATH = r'ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5'
STATS_CACHE_PATH = r'ascadv1_fixed_stats_cache.npy'

class ASCADv1_Fixed_Targets(Enum):
    SUBBYTES = 'subbytes'
    R = 'r'
    ROUT = 'r_out'
    SUBBYTES_XOR_R = 'subbytes_xor_r'
    SUBBYTES_XOR_ROUT = 'subbytes_xor_rout'
ASCADv1_Fixed_Targets_t = Union[ASCADv1_Fixed_Targets, str]

@dataclass
class _Database:
    traces: torch.Tensor
    plaintext: torch.Tensor
    ciphertext: torch.Tensor
    key: torch.Tensor
    masks: torch.Tensor
    store_in_ram: bool = False

    def __post_init__(self):
        assert isinstance(self.traces, torch.Tensor) and self.traces.dtype == torch.int8
        assert isinstance(self.plaintext, torch.Tensor) and self.plaintext.dtype == torch.uint8
        assert isinstance(self.ciphertext, torch.Tensor) and self.ciphertext.dtype == torch.uint8
        assert isinstance(self.key, torch.Tensor) and self.key.dtype == torch.uint8
        assert isinstance(self.masks, torch.Tensor) and self.masks.dtype == torch.uint8
        assert isinstance(self.store_in_ram, bool)
        assert len(self.traces) == len(self.plaintext) == len(self.ciphertext) == len(self.key) == len(self.masks)
        self.length = len(self.traces)
        if self.store_in_ram:
            self.traces.share_memory_()
            self.plaintext.share_memory_()
            self.ciphertext.share_memory_()
            self.key.share_memory_()
            self.masks.share_memory_()
    
    def __getitem__(self, idx: int):
        return self.traces[idx], self.plaintext[idx], self.ciphertext[idx], self.key[idx], self.masks[idx]

    def __len__(self) -> int:
        return self.length

_data: Optional[_Database] = None
def _init_shared_dataset(root: str):
    global _data
    if _data is not None:
        return
    with h5py.File(os.path.join(root, DATABASE_PATH), 'r') as f:
        traces = np.array(f['traces'])
        metadata_f = f['metadata']
        keys = list(metadata_f.dtype.fields.keys())
        metadata = {key: np.array(metadata_f[key], dtype=np.uint8) for key in keys}
    _data = _Database(traces=torch.from_numpy(traces), **{key: torch.from_numpy(val) for key, val in metadata.items()}, store_in_ram=True)

class ASCADv1_Fixed(Dataset):
    DATABASE_SHAPE = (60000, 100000)
    TRACE_SHAPE = (1, 100000)
    _barrier = None

    def __init__(self,
        root: str,
        train: bool = True,
        target: Union[ASCADv1_Fixed_Targets_t, List[ASCADv1_Fixed_Targets_t]] = ASCADv1_Fixed_Targets.SUBBYTES,
        store_in_ram: bool = False
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.target = target if isinstance(target, list) else [target]
        self.target = [ASCADv1_Fixed_Targets(x) if isinstance(x, str) else x for x in self.target]
        self.store_in_ram = store_in_ram
        self.database_path = os.path.join(self.root, DATABASE_PATH)
        self.needs_to_setup = True
        self.indices = np.arange(0, 50000) if self.train else np.arange(50000, 60000)
        if self.store_in_ram:
            _init_shared_dataset(self.root)
    
    def setup(self):
        global _data
        if self.store_in_ram:
            while _data is None:
                pass
            self.data = _data
        else:
            assert False
        self.aes_sbox = torch.from_numpy(AES_SBOX)
        self.mean, self.std = get_trace_sample_stats(self.data.traces, os.path.join(self.root, STATS_CACHE_PATH))
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
        if ASCADv1_Fixed_Targets.SUBBYTES in self.target:
            out.append(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)])
        if ASCADv1_Fixed_Targets.R in self.target:
            out.append(r.unsqueeze(0))
        if ASCADv1_Fixed_Targets.ROUT in self.target:
            out.append(r_out.unsqueeze(0))
        if ASCADv1_Fixed_Targets.SUBBYTES_XOR_R in self.target:
            out.append(torch.bitwise_xor(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)], r.unsqueeze(0)))
        if ASCADv1_Fixed_Targets.SUBBYTES_XOR_ROUT in self.target:
            out.append(torch.bitwise_xor(self.aes_sbox[torch.bitwise_xor(key, plaintext).to(torch.long)], r_out.unsqueeze(0)))
        assert len(out) > 0
        out = torch.cat(out)
        return out
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[idx]
        if self.needs_to_setup:
            self.setup()
        trace, plaintext, ciphertext, key, masks = self.data[idx]
        target = self.compute_target(key, plaintext, masks)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target
    
    def __len__(self) -> int:
        return len(self.indices)