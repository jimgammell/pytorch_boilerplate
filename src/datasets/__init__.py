import os
from typing import Tuple, Optional, Union
from enum import Enum

from torch.utils.data import Dataset

from common import *

class AVAILABLE_DATASETS(Enum):
    MNIST = 'mnist'
    ASCADv1_Fixed = 'ascadv1-fixed'
    ASCADv1_Var = 'ascadv1-var'
    ASCADv2 = 'ascadv2'
    IMAGENET = 'imagenet'
    IMAGENETTE = 'imagenette'
    OPENWEBTEXT = 'openwebtext'
    JESTER = 'jester'

def get_root(dataset_name: AVAILABLE_DATASETS) -> str:
    assert dataset_name in AVAILABLE_DATASETS
    if dataset_name == AVAILABLE_DATASETS.IMAGENET:
        return IMAGENET_ROOT
    elif dataset_name == AVAILABLE_DATASETS.IMAGENETTE:
        return IMAGENETTE_ROOT
    elif dataset_name == AVAILABLE_DATASETS.OPENWEBTEXT:
        return OPENWEBTEXT_ROOT
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1_Fixed:
        return ASCADv1_ROOT
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1_Var:
        return ASCADv1_ROOT
    elif dataset_name == AVAILABLE_DATASETS.ASCADv2:
        return ASCADv2_ROOT
    elif dataset_name == AVAILABLE_DATASETS.JESTER:
        return JESTER_ROOT
    else:
        return os.path.join(RESOURCE_DIR, dataset_name.value)

def load(dataset_name: Union[str, AVAILABLE_DATASETS], **kwargs) -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(dataset_name, str):
        dataset_name = AVAILABLE_DATASETS(dataset_name)
    root = get_root(dataset_name)
    assert root is not None
    if dataset_name == AVAILABLE_DATASETS.MNIST:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.IMAGENET:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.IMAGENETTE:
        from .imagenette import Imagenette
        train_dataset = Imagenette(root, train=True, **kwargs)
        test_dataset = Imagenette(root, train=False, **kwargs)
    elif dataset_name == AVAILABLE_DATASETS.OPENWEBTEXT:
        raise NotImplementedError
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1_Fixed:
        from .ascad.ascadv1 import ASCADv1_Fixed
        train_dataset = ASCADv1_Fixed(root, train=True, **kwargs)
        test_dataset = ASCADv1_Fixed(root, train=False, **kwargs)
    elif dataset_name == AVAILABLE_DATASETS.ASCADv1_Var:
        from .ascad.ascadv1 import ASCADv1_Var
        train_dataset = ASCADv1_Var(root, train=True, **kwargs)
        test_dataset = ASCADv1_Var(root, train=False, **kwargs)
    elif dataset_name == AVAILABLE_DATASETS.ASCADv2:
        from .ascad.ascadv2 import ASCADv2
        train_dataset = ASCADv2(root, train=True, **kwargs)
        test_dataset = None
    elif dataset_name == AVAILABLE_DATASETS.JESTER:
        from .jester import Jester, JesterDataModule
        train_dataset = Jester(JESTER_ROOT, split='train', **kwargs)
        val_dataset = Jester(JESTER_ROOT, split='validation', **kwargs)
        test_dataset = Jester(JESTER_ROOT, split='test', **kwargs)
        return train_dataset, val_dataset, test_dataset, JesterDataModule
    else:
        assert False
    return train_dataset, test_dataset