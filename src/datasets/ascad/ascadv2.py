import os
from enum import Enum
from typing import Optional, Callable, Union, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common import *
from utils.aes import AES_SBOX, GF256_LOG, GF256_ALOG
from ..base_dataset import BaseDataset
from ..utils import get_trace_sample_stats

G_PERM = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02], dtype=np.uint8)

class ASCADv2_Targets(Enum):
    SUBBYTES = 'subbytes' # just the SubBytes variable -- i.e. the fully-protected dataset
    ALPHA = 'alpha' # multiplicative mask
    BETA = 'beta' # additive mask
    PERM = 'perm' # permutation order
    SUBBYTES_ALPHA_MASKED = 'subbytes-alpha-masked' # alpha x SubBytes
    SUBBYTES_MASKED = 'subbytes-masked' # alpha x SubBytes + beta
    SUBBYTES_PERM = 'subbytes-perm' # SubBytes[perm]
    SUBBYTES_ALPHA_MASKED_PERM = 'subbytes-alpha-masked-perm' # alpha x SubBytes[perm]
    SUBBYTES_MASKED_PERM = 'subbytes-masked-perm' # alpha x SubBytes[perm] + beta
ASCADv2_Targets_t = Union[ASCADv2_Targets, str]

@dataclass
class _Database:
    traces: List[h5py.Dataset]
    plaintext: NDArray[np.uint8]
    key: NDArray[np.uint8]
    masks: NDArray[np.uint8]

    def __post_init__(self):
        assert all(isinstance(x, h5py.Dataset) and x.dtype == np.int8 for x in self.traces)
        assert isinstance(self.plaintext, np.ndarray) and self.plaintext.dtype == np.uint8
        assert isinstance(self.key, np.ndarray) and self.key.dtype == np.uint8
        assert isinstance(self.masks, np.ndarray) and self.masks.dtype == np.uint8
        self.datafile_lengths = [len(x) for x in self.traces]
        self.cumulative_datafile_lengths = np.cumsum([0, *self.datafile_lengths])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        database_idx = np.logical_and(idx >= self.cumulative_datafile_lengths[:-1], idx < self.cumulative_datafile_lengths[1:]).argmax()
        point_idx = idx - self.cumulative_datafile_lengths[database_idx]
        trace = torch.from_numpy(self.traces[database_idx][point_idx])
        return (trace, *list(map(lambda x: torch.from_numpy(x[idx]), (self.plaintext, self.key, self.masks))))
    
    def __len__(self) -> int:
        return sum(self.datafile_lengths)

class ASCADv2(Dataset):
    DATABASE_SHAPE = (800000, 1000000)
    TRACE_SHAPE = (1, 100000)
    PROFILE_INDICES = np.arange(800000)
    ATTACK_INDICES = np.array([], dtype=int)
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

    def __init__(self,
        root: str,
        train: bool = True,
        target: Union[ASCADv2_Targets_t, List[ASCADv2_Targets_t]] = ASCADv2_Targets.SUBBYTES
    ):
        super().__init__()

        self.root = root
        self.train = train
        assert self.train
        self.target = target if isinstance(target, list) else [target]
        self.target = [ASCADv2_Targets(x) if isinstance(x, str) else x for x in self.target]
        self.database_paths = [os.path.join(self.root, x) for x in self.DATABASE_FILENAMES]
        self.stats_cache_path = os.path.join(self.root, self.STATS_CACHE_FILENAME)
        self.needs_to_setup = True
        self.indices = self.PROFILE_INDICES if self.train else self.ATTACK_INDICES
    
    def setup(self):
        self.data_files = [
            h5py.File(x, 'r') for x in self.database_paths
        ]
        traces = [x['traces'] for x in self.data_files]
        plaintext = np.concatenate([np.array(x['metadata']['plaintext'], dtype=np.uint8) for x in self.data_files])
        key = np.concatenate([np.array(x['metadata']['key'], dtype=np.uint8) for x in self.data_files])
        masks = np.concatenate([np.array(x['metadata']['masks'], dtype=np.uint8) for x in self.data_files])
        self.data = _Database(traces=traces, plaintext=plaintext, key=key, masks=masks)
        self.aes_sbox = torch.from_numpy(AES_SBOX)
        self.g_perm = torch.from_numpy(G_PERM)
        self.gf256_log = torch.from_numpy(GF256_LOG)
        self.gf256_alog = torch.from_numpy(GF256_ALOG)
        self.mean, self.std = get_trace_sample_stats(traces[0], self.stats_cache_path)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float).reshape(*self.TRACE_SHAPE)),
            transforms.Lambda(lambda x: (x - self.mean)/self.std)
        ])
        self.target_transform = transforms.Lambda(lambda x: x.to(torch.long))
        self.needs_to_setup = False
    
    def gf256_prod(self, a: torch.Tensor, b: torch.Tensor):
        rv = torch.zeros_like(a)
        indices = ~torch.logical_or(a == 0, b == 0)
        rv[indices] = self.gf256_alog[((self.gf256_log[a.to(torch.long)] + self.gf256_log[b.to(torch.long)])%255).to(torch.long)]
        return rv

    def compute_target(self, key: torch.Tensor, plaintext: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        perm_params = masks[:4] & 0xFF
        perm_indices = torch.arange(16, dtype=torch.uint8) ^ 15
        for perm_param in perm_params:
            perm_indices = self.g_perm[(perm_indices ^ perm_param).to(torch.long)]
        alpha = masks[18].unsqueeze(0)
        beta = masks[17].unsqueeze(0)
        subbytes = self.aes_sbox[(key ^ plaintext).to(torch.long)]
        subbytes_perm = self.aes_sbox[(key[perm_indices] ^ plaintext[perm_indices]).to(torch.long)]
        subbytes_alpha = self.gf256_prod(torch.full_like(subbytes, alpha.item()), subbytes)
        subbytes_perm_alpha = self.gf256_prod(torch.full_like(subbytes_perm, alpha.item()), subbytes_perm)
        subbytes_masked = subbytes_alpha ^ beta.item()
        subbytes_perm_masked = subbytes_perm_alpha ^ beta.item()
        out = []
        if ASCADv2_Targets.SUBBYTES in self.target:
            out.append(subbytes)
        if ASCADv2_Targets.ALPHA in self.target:
            out.append(alpha)
        if ASCADv2_Targets.BETA in self.target:
            out.append(beta)
        if ASCADv2_Targets.SUBBYTES_PERM in self.target:
            out.append(subbytes_perm)
        if ASCADv2_Targets.SUBBYTES_ALPHA_MASKED in self.target:
            out.append(subbytes_alpha)
        if ASCADv2_Targets.SUBBYTES_ALPHA_MASKED_PERM in self.target:
            out.append(subbytes_perm_alpha)
        if ASCADv2_Targets.SUBBYTES_MASKED in self.target:
            out.append(subbytes_masked)
        if ASCADv2_Targets.SUBBYTES_MASKED_PERM in self.target:
            out.append(subbytes_perm_masked)
        assert len(out) > 0
        out = torch.cat(out)
        return out
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
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
        if hasattr(self, 'data_files'):
            for data_file in self.data_files:
                data_file.close()