import os
from typing import Literal, Optional, Dict, List, Union, Tuple
from random import uniform, randint
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info
import torchvision.transforms.v2 as tv_transforms
from torchvision.io import read_file, decode_image
import webdataset
from filelock import FileLock

from .prepare_data_files import *

class Jester(IterableDataset):
    def __init__(self, root: str, split: Literal['train', 'validation', 'test'] = 'train', dim: int = 128, timesteps: int = 8, prepare: bool = False):
        super().__init__()
        self.root = root
        self.split = split
        self.dim = dim
        self.timesteps = timesteps
        self.prepare = prepare

        if self.prepare:
            self.prepare_dataset()
        self.classes = load_classes(self.root)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float) + 1.e-6
        self.mean, self.std = map(lambda x: x.reshape(1, 3, 1, 1), (self.mean, self.std))
        if self.split == 'train':
            self.length = 118562
        elif self.split == 'validation':
            self.length = 14787
        elif self.split == 'test':
            self.length = 14743
        else:
            assert False

    def prepare_dataset(self):
        lock_path = os.path.join(self.root, 'lock')
        with FileLock(lock_path):
            download_data_files(self.root)
            extract_data_files(self.root)
            convert_to_webd_format(self.root)
    
    def extract_datapoint(self, sample):
        frame_indices = [int(key.split('.')[0]) for key in sample.keys() if key.endswith('.jpg')]
        frame_count = max(frame_indices)
        if frame_count > self.timesteps:
            if self.split == 'train':
                start_idx = randint(0, frame_count-self.timesteps)
            else:
                start_idx = (frame_count-self.timesteps)//2
            frame_indices = range(start_idx, start_idx+self.timesteps)
        elif frame_count < self.timesteps:
            frame_indices = list(range(frame_count)) + (self.timesteps-frame_count)*[frame_count-1]
        else:
            frame_indices = range(frame_count)
        video = torch.stack([sample[f'{idx+1:03d}.jpg'] for idx in frame_indices])
        if self.split == 'train':
            if randint(0, 1):
                video = video.flip(-1)
            size = int(uniform(1., 1.25)*self.dim)
            video = tv_transforms.functional.resize(video, size)
            start_row_idx = randint(0, size-self.dim)
            start_col_idx = randint(0, size-self.dim)
            video = video[:, :, start_row_idx:start_row_idx+self.dim, start_col_idx:start_col_idx+self.dim]
        else:
            video = tv_transforms.functional.resize(video, self.dim)
            video = tv_transforms.functional.center_crop(video, output_size=self.dim)
        if self.split == 'test':
            return video
        else:
            label = int(sample['cls'])
            return video, label

    def __iter__(self):
        shards_dir = os.path.join(self.root, f'{self.split}_shards')
        shard_indices = [int(filename.split('-')[1].split('.')[0]) for filename in os.listdir(shards_dir)]
        dataset = webdataset.DataPipeline(
            webdataset.SimpleShardList(os.path.join(shards_dir, r'shard-{'+f'{min(shard_indices):06d}..{max(shard_indices):06d}'+r'}.tar')),
            webdataset.split_by_worker,
            webdataset.detshuffle(),
            webdataset.tarfile_to_samples(),
            webdataset.shuffle(1000),
            webdataset.decode('torchrgb'),
            webdataset.map(self.extract_datapoint)
        ).with_length(self.length)
        return iter(dataset)