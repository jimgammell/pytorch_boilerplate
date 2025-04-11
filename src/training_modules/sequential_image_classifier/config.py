from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT

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
    compile: bool = False
    dtype: _PRECISION_INPUT = 'bf16-mixed'

    def __post_init__(self):
        pass