from typing import Dict, Any, Tuple
from dataclasses import dataclass

import torch
from torch import nn, optim
import lightning

import models
import utils.lr_schedulers as lr_schedulers
from .config import Config
from ..metrics import get_accuracy

@dataclass
class _ModuleHparams:
    classifier_name: str
    classifier_kwargs: Dict[str, Any]
    config: Config

class SupervisedVideoModule(lightning.LightningModule):
    hparams: _ModuleHparams

    def __init__(self,
        classifier_name: str,
        classifier_kwargs: Dict[str, Any],
        config: Config
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.load(self.hparams.classifier_name, self.hparams.classifier_kwargs)
        if self.hparams.config.compile:
            self.model.compile()
    
    def configure_optimizers(self):
        yes_weight_decay, no_weight_decay = self.model.get_params_based_on_should_weight_decay()
        param_groups = [
            {'params': yes_weight_decay, 'weight_decay': self.hparams.config.weight_decay},
            {'params': no_weight_decay, 'weight_decay': 0}
        ]
        optimizer = optim.AdamW(
            param_groups, lr=self.hparams.config.base_lr,
            betas=(self.hparams.config.beta_1, self.hparams.config.beta_2),
            eps=self.hparams.config.eps, fused=True
        )
        lr_scheduler = lr_schedulers.load(
            lr_schedulers.AVAILABLE_LR_SCHEDULERS(self.hparams.config.lr_scheduler_name),
            optimizer, int(self.trainer.estimated_stepping_batches), self.hparams.config.lr_scheduler_kwargs
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}}
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure = None):
        if self.hparams.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hparams.config.grad_clip)
        optimizer.step(optimizer_closure)
        optimizer.zero_grad()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, attn_mask, y = batch
        logits = self.model(x)
        batch_size, timesteps, class_count = logits.shape
        logits = logits.reshape(batch_size*timesteps, class_count)
        y = y.reshape(batch_size, 1).expand(-1, timesteps).reshape(batch_size*timesteps)
        loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.log('train_acc', get_accuracy(logits, y), prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, attn_mask, y = batch
        logits = self.model(x)
        batch_size, timesteps, class_count = logits.shape
        logits = logits.reshape(batch_size*timesteps, class_count)
        y = y.reshape(batch_size, 1).expand(-1, timesteps).reshape(batch_size*timesteps)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=False, on_epoch=True)
        self.log('val_acc', get_accuracy(logits, y), prog_bar=True, on_epoch=True)
        return loss