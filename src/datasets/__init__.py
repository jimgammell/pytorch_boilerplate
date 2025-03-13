import os
from typing import Tuple, Optional, Union
from enum import Enum

from torch.utils.data import Dataset

from common import *

class AVAILABLE_DATASETS(Enum):
    MNIST = 'mnist'
    ASCADv1 = 'ascadv1'
    ASCADv2 = 'ascadv2'
    IMAGENET = 'imagenet'
    OPENWEBTEXT = 'openwebtext'

def get_root(dataset_name: AVAILABLE_DATASETS) -> str:
    assert dataset_name in AVAILABLE_DATASETS
    if dataset_name == AVAILABLE_DATASETS.IMAGENET:
        return IMAGENET_ROOT
    elif dataset_name == AVAILABLE_DATASETS.OPENWEBTEXT:
        return OPENWEBTEXT_ROOT
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1:
        return ASCADv1_ROOT
    elif dataset_name == AVAILABLE_DATASETS.ASCADv2:
        return ASCADv2_ROOT
    else:
        return os.path.join(RESOURCE_DIR, dataset_name.value)

def load(dataset_name: Union[str, AVAILABLE_DATASETS], **kwargs) -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(dataset_name, str):
        dataset_name = AVAILABLE_DATASETS(dataset_name)
    root = get_root(dataset_name)
    if dataset_name == AVAILABLE_DATASETS.MNIST:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.IMAGENET:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.OPENWEBTEXT:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.ASCADv2:
        from .ascad.ascadv2 import ASCADv2
        train_dataset = ASCADv2(root, **kwargs)
        test_dataset = None
    else:
        assert False
    return train_dataset, test_dataset