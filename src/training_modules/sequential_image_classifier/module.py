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

class SequentialImageClassifierModule(lightning.LightningModule):
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
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self.model.single_training_step(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=False, on_step=True)
        self.log('train_acc', get_accuracy(logits, y), prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits_from_patch_predictor, _ = self.model.run_inference(x, max_iters=self.hparams.classifier_kwargs['max_sequence_length'])
        logits_from_random_sequence = self.model.run_inference_with_random_sequence(x, max_iters=self.hparams.classifier_kwargs['max_sequence_length'])
        batch_size, seq_length, class_count = logits_from_patch_predictor.shape
        assert logits_from_patch_predictor.shape == logits_from_random_sequence.shape
        yy = y.unsqueeze(1).expand(batch_size, seq_length)
        loss_patch_predictor = nn.functional.cross_entropy(
            logits_from_patch_predictor.reshape(batch_size*seq_length, class_count), yy.reshape(batch_size*seq_length), reduction='none'
        ).reshape(batch_size, seq_length).mean(dim=0)
        loss_random_sequence = nn.functional.cross_entropy(
            logits_from_random_sequence.reshape(batch_size*seq_length, class_count), yy.reshape(batch_size*seq_length), reduction='none'
        ).reshape(batch_size, seq_length).mean(dim=0)
        acc_patch_predictor = get_accuracy(
            logits_from_patch_predictor.reshape(batch_size*seq_length, class_count), yy.reshape(batch_size*seq_length), avg_result=False
        ).reshape(batch_size, seq_length).mean(dim=0)
        acc_random_sequence = get_accuracy(
            logits_from_random_sequence.reshape(batch_size*seq_length, class_count), yy.reshape(batch_size*seq_length), avg_result=False
        ).reshape(batch_size, seq_length).mean(dim=0)
        self.log('val_loss', loss_patch_predictor.mean(), prog_bar=False, on_epoch=True)
        self.log('val_loss_baseline', loss_random_sequence.mean(), prog_bar=False, on_epoch=True)
        self.log('val_acc', acc_patch_predictor.mean(), prog_bar=False, on_epoch=True)
        self.log('val_acc_baseline', acc_random_sequence.mean(), prog_bar=False, on_epoch=True)
        self.log('final_val_loss', loss_patch_predictor[-1], prog_bar=False, on_epoch=True)
        self.log('final_val_loss_baseline', loss_random_sequence[-1], prog_bar=False, on_epoch=True)
        self.log('final_val_acc', acc_patch_predictor[-1], prog_bar=True, on_epoch=True)
        self.log('final_val_acc_baseline', acc_random_sequence[-1], prog_bar=True, on_epoch=True)
        return loss_patch_predictor.mean()