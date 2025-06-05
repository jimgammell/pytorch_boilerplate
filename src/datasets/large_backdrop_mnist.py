from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

class LargeBackdropMNIST(Dataset):
    def __init__(self, root: str, stage: Literal['train', 'test'] = 'train', mnist_dim: int = 8, background_dim: int = 64, timesteps: int = 8):
        super().__init__()
        self.root = root
        self.stage = stage
        self.mnist_dim = mnist_dim
        self.background_dim = background_dim
        self.video_len = timesteps
        self.dataset = None
    
    def construct_dataset(self):
        self.data_transform = transforms.Compose([
            transforms.Resize(self.mnist_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        self.dataset = MNIST(root=self.root, train=self.stage=='train', transform=self.data_transform, download=True)

    def sample_image(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        background = torch.zeros(1, self.background_dim, self.background_dim, dtype=image.dtype, device=image.device)
        start_row, start_col = np.random.randint(0, self.background_dim-self.mnist_dim, size=2)
        background[:, start_row:start_row+self.mnist_dim, start_col:start_col+self.mnist_dim] = image
        background = background.unsqueeze(0).repeat(self.video_len, 1, 1, 1)
        return background, label
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.dataset is None:
            self.construct_dataset()
        image, label = self.sample_image(idx)
        return image, label
    
    def __len__(self) -> int:
        if self.dataset is None:
            self.construct_dataset()
        return len(self.dataset)