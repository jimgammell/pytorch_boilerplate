from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import lightning

from common import *
from .base_dataset import BaseDataset

class RepeatedAugmentationSampler(Sampler):
    def __init__(self, data_source, num_repeats=3, shuffle=True):
        self.data_source = data_source
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.sample_count = len(self.data_source)*self.num_repeats
    
    def __iter__(self):
        indices = np.arange(len(self.data_source))
        if self.shuffle:
            np.random.shuffle(indices)
        indices = indices.repeat(self.num_repeats)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.sample_count

@dataclass
class DataModuleConfig:
    val_prop: float = 0.1
    train_batch_size: int = 256
    eval_batch_size: int = 2048
    num_workers: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    def __post_init__(self):
        assert isinstance(self.val_prop, float)
        assert isinstance(self.train_batch_size, int)
        assert isinstance(self.eval_batch_size, int)
        assert self.num_workers is None or isinstance(self.num_workers, int)
        assert isinstance(self.pin_memory, bool)
        assert isinstance(self.persistent_workers, bool)
        assert isinstance(self.prefetch_factor, int)
        assert 0 <= self.val_prop < 1
        assert 0 < self.train_batch_size
        assert 0 < self.eval_batch_size
        assert self.num_workers is None or 0 <= self.num_workers
        assert 0 <= self.prefetch_factor

class DataModule(lightning.LightningDataModule):
    def __init__(self,
        base_train_dataset: BaseDataset,
        config: DataModuleConfig,
        test_dataset: Optional[BaseDataset],
    ):
        super().__init__()
        self.base_train_dataset = base_train_dataset
        self.config = config
        self.test_dataset = test_dataset
        self.indices = None
        self.setup('')
    
    def setup(self, stage: str):
        self.val_length = int(len(self.base_train_dataset)*self.config.val_prop)
        if self.indices is None:
            self.indices = np.random.choice(len(self.base_train_dataset), len(self.base_train_dataset), replace=False).astype(int).tolist()
        train_indices = self.indices[self.val_length:]
        val_indices = self.indices[:self.val_length]
        self.train_dataset = Subset(self.base_train_dataset, train_indices)
        self.val_dataset = Subset(self.base_train_dataset, val_indices)
        if hasattr(self.base_train_dataset, 'enable_data_transforms'):
            self.train_dataset.dataset.enable_data_transforms(set=True, aug=True)
            self.val_dataset.dataset.enable_data_transforms(set=True, aug=False)
        self.dataloader_kwargs: Dict[str, Any] = dict(
            num_workers = self.config.num_workers or get_worker_count(),
            pin_memory = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor = self.config.prefetch_factor
        )
        self.repeat_augment_sampler = RepeatedAugmentationSampler(self.train_dataset)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, sampler=self.repeat_augment_sampler, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval_batch_size, **self.dataloader_kwargs)

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, batch_size=self.config.eval_batch_size, **self.dataloader_kwargs)
    
    def on_load_checkpoint(self, checkpoint):
        self.indices = checkpoint.get('indices', None)