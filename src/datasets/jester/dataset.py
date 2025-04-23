import os
from typing import Literal, Optional, Dict, List, Union, Tuple
from random import uniform, randint

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tv_transforms
from torchvision.io import read_file, decode_image

class Jester(Dataset):
    classes: Optional[Dict[str, int]] = None

    @staticmethod
    def split_agnostic_init(root: str):
        if Jester.classes is None:
            label_path = os.path.join(root, 'jester-v1-labels.csv')
            with open(label_path, 'r') as f:
                Jester.classes = {
                    line.strip(): idx for idx, line in enumerate(f)
                }

    def __init__(self, 
        root: str, split: Literal['train', 'val', 'test'] = 'train', dim: int = 128, timesteps: int = 8
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.dim = dim
        self.timesteps = timesteps
        self.split_agnostic_init(self.root)
        if self.split in ['train', 'val']:
            if self.split == 'train':
                labels_file = os.path.join(root, 'jester-v1-train.csv')
            elif self.split == 'val':
                labels_file = os.path.join(root, 'jester-v1-validation.csv')
            self.data_indices, self.data_labels = [], []
            with open(labels_file, 'r') as f:
                for line in f:
                    data_idx, data_label = line.strip().split(';')
                    self.data_indices.append(int(data_idx))
                    self.data_labels.append(Jester.classes[data_label])
            self.data_indices = np.array(self.data_indices)
            self.data_labels = torch.tensor(self.data_labels, dtype=torch.long)
        elif self.split == 'test':
            labels_file = os.path.join(self.root, 'jester-v1-test.csv')
            self.data_indices = []
            self.data_labels = None
            with open(labels_file, 'r') as f:
                for line in f:
                    data_idx = int(line.strip())
                    self.data_indices.append(data_idx)
            self.data_indices = np.array(self.data_indices)
        else:
            assert False, self.split
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float) + 1.e-6
        self.mean, self.std = map(lambda x: x.reshape(1, 3, 1, 1), (self.mean, self.std))
        self.paths = {}
        for idx in self.data_indices:
            subdir = os.path.join(self.root, '20bn-jester-v1', f'{idx}')
            frame_indices = [int(x.split('.')[0]) for x in os.listdir(subdir) if x.endswith('.jpg')]
            frame_indices.sort()
            self.paths[idx] = [os.path.join(subdir, f'{frame_idx:05d}.jpg') for frame_idx in frame_indices]
    
    def load_video(self, idx: int) -> torch.Tensor:
        video = []
        paths = self.paths[idx]
        if len(paths) > self.timesteps:
            start_idx = randint(0, len(paths)-self.timesteps)
            paths = paths[start_idx:start_idx+self.timesteps]
        elif len(paths) < self.timesteps:
            paths = [*paths, *((self.timesteps-len(paths))*[paths[-1]])]
        for frame_path in paths:
            frame = decode_image(read_file(frame_path), mode='RGB')
            video.append(frame)
        video = torch.stack(video)
        return video
    
    def transform_video(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float().div_(255)
        video = video.sub_(self.mean).div_(self.std)
        if self.split == 'train': # I'm implementing the transforms manually -- torchvision version doesn't support the same random seed for all frames
            # random horizontal flip
            if randint(0, 1):
                video = video.flip(-1)
            # random resize and crop
            size = int(uniform(1., 1.25)*self.dim)
            video = tv_transforms.functional.resize(video, size)
            start_row_idx = randint(0, size-self.dim)
            start_col_idx = randint(0, size-self.dim)
            video = video[:, :, start_row_idx:start_row_idx+self.dim, start_col_idx:start_col_idx+self.dim]
            # color jitter with brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, according to ChatGPT
            b = uniform(0.6, 1.4)
            c = uniform(0.6, 1.4)
            s = uniform(0.6, 1.4)
            h = uniform(-0.1, 0.1)
            frames = []
            for frame in video:
                frame = tv_transforms.functional.adjust_brightness(frame, b)
                frame = tv_transforms.functional.adjust_contrast(frame, c)
                frame = tv_transforms.functional.adjust_saturation(frame, s)
                frame = tv_transforms.functional.adjust_hue(frame, h)
                frames.append(frame)
            video = torch.stack(frames)
        else:
            video = tv_transforms.functional.resize(video, self.dim)
            video = tv_transforms.functional.center_crop(video, output_size=self.dim)
        return video

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        data_idx = self.data_indices[idx]
        video = self.load_video(data_idx)
        video = self.transform_video(video)
        if self.data_labels is not None:
            label = self.data_labels[idx]
            return video, label
        else:
            return video
    
    def __len__(self) -> int:
        return len(self.data_indices)