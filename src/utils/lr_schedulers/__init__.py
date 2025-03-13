from typing import Dict, Any
from enum import Enum

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class AVAILABLE_LR_SCHEDULERS(Enum):
    COSINE_DECAY = 'cosine_decay'

def load(name: AVAILABLE_LR_SCHEDULERS, optimizer: Optimizer, total_steps: int, config: Dict[str, Any]) -> LRScheduler:
    if name == AVAILABLE_LR_SCHEDULERS.COSINE_DECAY:
        from .cosine_decay import CosineDecayLRSched
        return CosineDecayLRSched(optimizer, total_steps, **config)
    else:
        assert False