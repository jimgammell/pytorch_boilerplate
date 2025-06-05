from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import os

from models.autoregressive_diffusion.arm_building_blocks import Config as ARMConfig, GPT2_DEFAULT_KWARGS

@dataclass
class Config:
    pretrained_arm_path: Optional[str] = None
    arm_kwargs: Dict[str, Any] = {}
    pretrained_gpt2_size: Optional[Literal['s', 'm', 'l', 'xl']] = None

    def __post_init__(self):
        if self.pretrained_gpt2_size is not None:
            override_kwargs = GPT2_DEFAULT_KWARGS[self.pretrained_gpt2_size]
            assert all(not(k in self.arm_kwargs) or (self.arm_kwargs[k] == v) for k, v in override_kwargs.items())
            self.arm_kwargs.update(override_kwargs)
        self.arm_config = ARMConfig(**self.arm_kwargs)
        del self.arm_kwargs
        if self.pretrained_arm_path is not None:
            assert os.path.exists(self.pretrained_arm_path)