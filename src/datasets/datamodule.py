from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import lightning

from common import *

@dataclass
class DataModuleConfig:
    val_prop: float = 0.1
    train_batch_size: int = 256
    eval_batch_size: int = 2048
    num_workers: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

class DataModule(lightning.LightningDataModule):
    def __init__(self,
        base_train_dataset: Dataset,
        config: DataModuleConfig,
        test_dataset: Optional[Dataset],
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
            self.indices = np.random.choice(len(self.base_train_dataset), len(self.base_train_dataset), replace=True)
        train_indices = self.indices[self.val_length:]
        val_indices = self.indices[:self.val_length]
        self.train_dataset = Subset(self.base_train_dataset, train_indices)
        self.val_dataset = Subset(self.base_train_dataset, val_indices)
        self.dataloader_kwargs: Dict[str, Any] = dict(
            num_workers = self.config.num_workers or get_worker_count(),
            pin_memory = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor = self.config.prefetch_factor
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval_batch_size, **self.dataloader_kwargs)

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, batch_size=self.config.eval_batch_size, **self.dataloader_kwargs)
    
    def on_load_checkpoint(self, checkpoint):
        self.indices = checkpoint.get('indices', None)