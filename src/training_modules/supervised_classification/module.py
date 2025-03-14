from typing import Dict, Any, Tuple
from dataclasses import dataclass

import torch
from torch import nn, optim
import lightning

import utils.lr_schedulers as lr_schedulers
import models
from .config import SupervisedClassificationConfig
from ..metrics import get_accuracy, get_rank

@dataclass
class _ModuleHparams:
    classifier_name: models.AVAILABLE_MODELS
    classifier_kwargs: Dict[str, Any]
    training_config: SupervisedClassificationConfig

class SupervisedClassificationModule(lightning.LightningModule):
    hparams: _ModuleHparams

    def __init__(self,
        classifier_name: str,
        classifier_kwargs: Dict[str, Any],
        training_config: SupervisedClassificationConfig
    ):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = models.load(self.hparams.classifier_name, self.hparams.classifier_kwargs)
        if self.hparams.training_config.compile:
            self.classifier.compile()
        
    def configure_optimizers(self):
        yes_weight_decay, no_weight_decay = self.classifier.get_params_based_on_should_weight_decay()
        param_groups = [
            {'params': yes_weight_decay, 'weight_decay': self.hparams.training_config.weight_decay},
            {'params': no_weight_decay, 'weight_decay': 0}
        ]
        optimizer = optim.AdamW(
            param_groups, lr=self.hparams.training_config.base_lr,
            betas=(self.hparams.training_config.beta_1, self.hparams.training_config.beta_2),
            eps=self.hparams.training_config.eps, fused=True
        )
        lr_scheduler = lr_schedulers.load(
            lr_schedulers.AVAILABLE_LR_SCHEDULERS(self.hparams.training_config.lr_scheduler_name),
            optimizer, int(self.trainer.estimated_stepping_batches), self.hparams.training_config.lr_scheduler_kwargs
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
        if self.hparams.training_config.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=self.hparams.training_config.grad_clip)
        optimizer.step(optimizer_closure)
        optimizer.zero_grad()
    
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, log_prefix: str = ''):
        x, y = batch
        logits = self.classifier(x)
        if logits.dim() == 2: # single task learning
            batch_size, class_count = logits.shape
            assert (batch_size,) == y.shape
            loss = nn.functional.cross_entropy(logits, y)
        elif logits.dim() == 3: # multitask learning
            batch_size, task_count, class_count = logits.shape
            assert (batch_size, task_count) == y.shape
            #logits = logits.reshape(batch_size*task_count, class_count)
            #y = y.reshape(batch_size*task_count)
            logits = logits[:, 0, :]
            y = y[:, 0]
            loss = nn.functional.cross_entropy(logits, y)
        else:
            assert False
        self.log(f'{log_prefix}_loss', loss, prog_bar=True, on_step=True)
        self.log(f'{log_prefix}_acc', get_accuracy(logits, y), prog_bar=False, on_epoch=True)
        self.log(f'{log_prefix}_rank', get_rank(logits, y), prog_bar=True, on_epoch=True)
        return loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self.step(batch, batch_idx, log_prefix='train')
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self.step(batch, batch_idx, log_prefix='val')
        return loss