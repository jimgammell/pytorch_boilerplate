from typing import Any, Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray

from torch.utils.data import Dataset

class BaseDataset(Dataset, metaclass=ABCMeta):
    @property
    @abstractmethod
    def mean(self) -> NDArray[np.float32]:
        raise NotImplementedError
    @property
    @abstractmethod
    def std(self) -> NDArray[np.float32]:
        raise NotImplementedError
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError
    def __len__(self) -> int:
        raise NotImplementedError