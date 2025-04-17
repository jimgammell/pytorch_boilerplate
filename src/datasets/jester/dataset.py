import os
from typing import Literal, Optional, Dict, List, Union, Tuple

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

class Jester(Dataset):
    classes: Optional[Dict[str, int]] = None
    videos: Optional[Dict[int, List[torch.Tensor]]] = None

    @staticmethod
    def load_data_into_ram(root: str):
        if False: #Jester.videos is None:
            Jester.videos = {}
            base_dir = os.path.join(root, '20bn-jester-v1')
            video_indices = [int(x) for x in os.listdir(base_dir)]
            for video_index in tqdm(video_indices):
                subdir = os.path.join(base_dir, f'{video_index}')
                frame_indices = [int(x.split('.')[0]) for x in os.listdir(subdir) if x.endswith('.jpg')]
                frame_indices.sort()
                video_frames = [
                    torch.from_numpy(np.array(Image.open(os.path.join(subdir, f'{frame_index:05d}.jpg')))).permute(2, 0, 1).contiguous()
                    for frame_index in frame_indices
                ]
                video_tensor = torch.stack(video_frames)
                video_tensor.share_memory_()
                Jester.videos[video_index] = video_tensor
        if Jester.classes is None:
            label_path = os.path.join(root, 'jester-v1-labels.csv')
            with open(label_path, 'r') as f:
                Jester.classes = {
                    line.strip(): idx for idx, line in enumerate(f)
                }

    def __init__(self, 
        root: str, split: Literal['train', 'val', 'test'] = 'train', dim: int = 128
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.dim = dim
        self.load_data_into_ram(self.root)
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
            self.data_indices = torch.tensor(self.data_indices, dtype=torch.long)
            self.data_labels = torch.tensor(self.data_labels, dtype=torch.long)
        elif self.split == 'test':
            labels_file = os.path.join(self.root, 'jester-v1-test.csv')
            self.data_indices = []
            self.data_labels = None
            with open(labels_file, 'r') as f:
                for line in f:
                    data_idx = int(line.strip())
                    self.data_indices.append(data_idx)
            self.data_indices = torch.tensor(self.data_indices, dtype=torch.long)
        else:
            assert False, self.split
        self.data_transform = tv_transforms.Compose([
            tv_transforms.Lambda(lambda x: x.to(torch.float)),
            tv_transforms.Resize(self.dim, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(self.dim),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_video(self, idx: int) -> torch.Tensor:
        subdir = os.path.join(self.root, '20bn-jester-v1', f'{idx}')
        frame_indices = [int(x.split('.')[0]) for x in os.listdir(subdir) if x.endswith('.jpg')]
        frame_indices.sort()
        video_frames = []
        for frame_index in frame_indices:
            with Image.open(os.path.join(subdir, f'{frame_index:05d}.jpg')) as img:
                img = img.convert('RGB')
                img = tv_transforms.functional.to_tensor(img)
                video_frames.append(img)
        video_tensor = torch.stack(video_frames)
        return video_tensor

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        data_idx = self.data_indices[idx]
        video = self.load_video(data_idx)
        video = self.data_transform(video)
        if self.data_labels is not None:
            label = self.data_labels[idx]
            return video, label
        else:
            return video
    
    def __len__(self) -> int:
        return len(self.data_indices)