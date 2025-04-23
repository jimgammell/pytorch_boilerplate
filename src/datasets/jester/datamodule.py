from typing import Optional, Dict, Any, Tuple, Sequence, Union
from random import randint
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from common import *

@dataclass
class DataModuleConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 1
    timestep_count: int = 32

    def __post_init__(self):
        assert isinstance(self.train_batch_size, int) and (self.train_batch_size > 0)
        assert isinstance(self.eval_batch_size, int) and (self.eval_batch_size > 0)
        if self.num_workers is None:
            self.num_workers = get_worker_count()
        assert isinstance(self.num_workers, int) and (self.num_workers >= 0)
        assert isinstance(self.pin_memory, bool)
        assert isinstance(self.persistent_workers, bool)
        if self.prefetch_factor is not None:
            assert isinstance(self.prefetch_factor, int) and (self.prefetch_factor > 0)
        assert isinstance(self.timestep_count, int) and (self.timestep_count > 0)

class JesterDataModule(LightningDataModule):
    def __init__(self,
        train_dataset: Dataset, val_dataset: Dataset, test_dataset: Optional[Dataset] = None, kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        kwargs = kwargs or {}
        self.config = DataModuleConfig(**kwargs)
    
    # We need all videos to have the same timestep count so we can stack into batches and use torch.compile.
    #  I am randomly clipping videos which are too long, and padding videos which are too short.
    #  This returns an attention mask so we can ignore padding.
    def collate(self, batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        videos, labels = zip(*batch)
        batch_size = len(videos)
        assert batch_size == len(labels)
        _, channels, height, width = videos[0].shape
        assert all((video.shape[1] == channels) and (video.shape[2] == height) and (video.shape[3] == width) for video in videos)
        padded_videos = torch.zeros((batch_size, self.config.timestep_count, channels, height, width), dtype=videos[0].dtype)
        attn_masks = torch.zeros((batch_size, self.config.timestep_count), dtype=torch.bool)
        for idx, video in enumerate(videos):
            if video.shape[0] > self.config.timestep_count:
                start = randint(0, video.shape[0]-self.config.timestep_count)
                padded_videos[idx, ...] = video[start:start+self.config.timestep_count, ...]
                attn_masks[idx, ...] = 1
            else:
                padded_videos[idx, :video.shape[0], ...] = video
                attn_masks[idx, :video.shape[0], ...] = 1
        labels = torch.stack(labels)
        return padded_videos, labels

    def setup(self, **kwargs):
        self.dataloader_kwargs: Dict[str, Any] = dict(
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.collate
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval_batch_size, shuffle=False, **self.dataloader_kwargs)
    
    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, batch_size=self.config.eval_batch_size, shuffle=False, **self.dataloader_kwargs)