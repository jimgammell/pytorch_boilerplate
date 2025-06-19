from dataclasses import dataclass
from typing import Dict, Any, Sequence

import torch
from torch import optim, nn
import lightning

import models
from models.autoregressive_diffusion import ARM, ARMConfig

@dataclass
class TrainingConfig:
    compile: bool = False

class Hparams:
    model_config: Dict[str, Any]
    training_config: TrainingConfig
    sigma_min: float = 0.002
    sigma_max: float = 80.
    sigma_data: float = 0.5
    rho: float = 7.
    p_mean: float = -1.2
    p_std: float = 1.2

class TrainingModule(lightning.LightningModule):
    hparams: Hparams

    def __init__(self,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.autoregressive_model = ARM(ARMConfig(**self.hparams.model_config))
        if self.hparams.training_config.compile:
            self.autoregressive_model.compile()