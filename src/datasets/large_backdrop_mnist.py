from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

class LargeBackdropMNIST(Dataset):
    def __init__(self, root: str, stage: Literal['train', 'test'] = 'train', mnist_dim: int = 8, background_dim: int = 64, video_len: int = 8):
        super().__init__()
        self.root = root
        self.stage = stage
        self.mnist_dim = mnist_dim
        self.background_dim = background_dim
        self.video_len = video_len
    
    def construct_dataset(self):
        self.data_transform = transforms.Compose([
            transforms.Resize(self.mnist_dim, self.mnist_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        self.dataset = MNIST(root=self.root, train=self.stage=='train', transform=self.data_transform)
        if self.stage == 'test':
            self.corner_indices = np.random.randint()

    def sample_image(self, idx: int) -> torch.Tensor:
        pass