# Based on comments here: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16

import h5py
import numpy as np
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, keys, transforms=None):
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)
        super().__init__()
        self.hdf5_file = None
        if self.transforms is None:
            self.transforms = len(self.keys)*[None]
        self.length = None
        with open(self.hdf5_path, 'r') as hdf5_file:
            for key in self.keys():
                if self.length is None:
                    self.length = hdf5_file[key].shape[0]
                else:
                    assert self.length == hdf5_file[key].shape[0]
    
    def get_hdf5_file(self): # Dataset will no longer be serializable after this is called.
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self.hdf5_file

    def __getitem__(self, indices):
        hdf5_file = self.get_hdf5_file()
        items = []
        for key, transform in zip(self.keys, self.transforms):
            item = np.array(hdf5_file[key][indices, ...])
            if transform is not None:
                item = transform(item)
            items.append(item)
        return items
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()