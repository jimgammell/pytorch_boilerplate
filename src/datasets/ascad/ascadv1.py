import os

import h5py
import torch
from torch.utils.data import Dataset

class ASCADv1(Dataset):
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        self.root = root
        self.train = train
        self.database = None
    
    def __getitem__(self):
        if self.database is None:
            self.database = h5py.File(os.path.join(self.root, r'atmega8515-raw-traces.h5'))