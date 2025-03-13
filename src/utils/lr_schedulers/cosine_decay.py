from math import isfinite

import numpy as np
from torch import optim

class CosineDecayLRSched(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: optim.Optimizer, total_steps: int, warmup_steps: int = 0, const_steps: int = 0, final_lr_prop: float = 1e-1):
        assert isinstance(optimizer, optim.Optimizer)
        assert isinstance(total_steps, int)
        assert isinstance(warmup_steps, int)
        assert isinstance(const_steps, int)
        assert isinstance(final_lr_prop, float)
        assert 0 < total_steps
        assert 0 <= warmup_steps <= total_steps
        assert 0 <= const_steps <= total_steps
        assert 0 <= final_lr_prop <= 1
        assert warmup_steps + const_steps <= total_steps
        self.total_steps = total_steps
        decay_steps = self.total_steps - (warmup_steps + const_steps)
        self.scheduler = np.concatenate([
            np.linspace(0, 1, warmup_steps),
            np.ones(const_steps),
            (1 - final_lr_prop)*(0.5*np.cos(np.linspace(0, np.pi, decay_steps)) + 0.5) + final_lr_prop
        ])
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, current_step: int) -> float:
        assert 0 <= current_step < self.total_steps
        return self.scheduler[current_step]