from typing import Dict, Any, Optional
from dataclasses import dataclass
from math import isfinite

@dataclass
class SupervisedClassificationConfig:
    base_lr: float = 2e-4
    lr_scheduler_name: Optional[str] = None
    lr_scheduler_kwargs: Dict[str, Any] = {}
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    def __post_init__(self):
        assert isinstance(self.base_lr, float)
        assert self.lr_scheduler_name is None or isinstance(self.lr_scheduler_name, str)
        assert isinstance(self.lr_scheduler_kwargs, dict)
        assert isinstance(self.beta_1, float)
        assert isinstance(self.beta_2, float)
        assert isinstance(self.eps, float)
        assert isinstance(self.weight_decay, float)
        assert self.grad_clip is None or isinstance(self.grad_clip, float)
        assert 0 < self.base_lr and isfinite(self.base_lr)
        assert all(isinstance(x, str) for x in self.lr_scheduler_kwargs)
        assert 0 < self.beta_1 < 1
        assert 0 < self.beta_2 < 1
        assert 0 < self.eps and isfinite(self.eps)
        assert 0 <= self.weight_decay and isfinite(self.weight_decay)
        assert self.grad_clip is None or (0 <= self.grad_clip and isfinite(self.grad_clip))