import os
from enum import Enum
from typing import Optional, Callable, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.aes import AES_SBOX, gf256_prod

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

class ASCADv2_Targets(Enum):
    UNPROTECTED = 'unprotected' # masking and permutation disabled
    MASKED = 'masked' # permutation disabled
    PERMUTED = 'permuted' # masking disabled
    FULL = 'full' # fully-protected ASCADv2 implementation

class ASCADv2(Dataset):
    def __init__(self, root: str, target: Union[str, ASCADv2_Targets], transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        self.target = target if isinstance(target, ASCADv2_Targets) else ASCADv2_Targets(target)
        self.databases = None
        self.transform = transform or transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float))
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
            out = gf256_prod(alpha, out) ^ beta
        return out

    def load_datapoint(self, idx):
        if self.databases is None: # init lazily to avoid multiprocessing-related issues
            self.databases = [
                h5py.File(os.path.join(self.root, filename), 'r', swmr=True) for filename in DATABASE_FILENAMES
            ]
            self.database_lengths = [len(x) for x in self.databases]
            self.cumulative_database_lengths = np.cumsum([0] + self.database_lengths)
        database_idx = np.logical_and(idx >= self.cumulative_database_lengths[:-1], idx < self.cumulative_database_lengths[1:]).argmax()
        point_idx = idx - self.cumulative_database_lengths[database_idx]
        trace = np.array(self.databases[database_idx]['traces'][point_idx], dtype=np.int8)
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