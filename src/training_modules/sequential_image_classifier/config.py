from typing import Optional, Dict, Any, get_args
from dataclasses import dataclass, field
from math import isfinite

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from utils.lr_schedulers import AVAILABLE_LR_SCHEDULERS

@dataclass
class Config:
    training_steps: int = 10000
    base_lr: float = 1e-4
    beta_1: float = 0.9
    beta_2: float = 0.99
    weight_decay: float = 1e-2
    eps: float = 1e-8
    grad_clip: Optional[float] = 1.0
    lr_scheduler_name: Optional[str] = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_mixup_and_cutmix: bool = True
    pretrain: bool = False
    compile: bool = False
    dtype: _PRECISION_INPUT = 'bf16-mixed'

    def __post_init__(self):
        assert isinstance(self.training_steps, int) and (self.training_steps > 0)
        assert isinstance(self.base_lr, float) and (self.base_lr > 0) and isfinite(self.base_lr)
        assert isinstance(self.beta_1, float) and (0 <= self.beta_1 < 1)
        assert isinstance(self.beta_2, float) and (0 < self.beta_2 < 1)
        assert isinstance(self.weight_decay, float) and (self.weight_decay >= 0) and isfinite(self.weight_decay)
        assert isinstance(self.eps, float) and (self.eps > 0) and isfinite(self.eps)
        if self.grad_clip is not None:
            assert isinstance(self.grad_clip, float) and (self.grad_clip > 0) and isfinite(self.grad_clip)
        assert (self.lr_scheduler_name is None) or (isinstance(self.lr_scheduler_name, str) and any(x.value == self.lr_scheduler_name for x in AVAILABLE_LR_SCHEDULERS))
        assert isinstance(self.lr_scheduler_kwargs, dict) and all(isinstance(x, str) for x in self.lr_scheduler_kwargs.keys())
        assert isinstance(self.use_mixup_and_cutmix, bool)
        assert isinstance(self.pretrain, bool)
        assert isinstance(self.compile, bool)
        assert isinstance(self.dtype, str) and (self.dtype in get_args(x) for x in get_args(_PRECISION_INPUT))