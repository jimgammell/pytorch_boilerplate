from typing import Dict, Any, Tuple
from dataclasses import dataclass
from random import randint
from itertools import chain

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
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
        if self.model.config.pretrained_lightning_module_path is not None:
            mod = SupervisedVideoModule.load_from_checkpoint(self.model.config.pretrained_lightning_module_path)
            pretrain_state_dict = mod.model.state_dict()
            self.model.load_state_dict({k: v for k, v in pretrain_state_dict.items() if k != 'head.attention_pool.q'}, strict=False)
        self.automatic_optimization = not self.model.config.sparse_inputs
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
    
    def standard_training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        batch_size, timesteps, class_count = logits.shape
        logits = logits.reshape(batch_size*timesteps, class_count)
        y = y.reshape(batch_size, 1).expand(-1, timesteps).reshape(batch_size*timesteps)
        loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1, reduction='none').reshape(batch_size, timesteps).mean(dim=0)
        acc = get_accuracy(logits, y, avg_result=False).reshape(batch_size, timesteps).mean(dim=0)
        self.log('train_loss', loss.mean(), prog_bar=False, on_step=True)
        self.log('train_loss_final', loss[-1], prog_bar=True, on_step=True)
        self.log('train_acc', acc.mean(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_acc_final', acc[-1], prog_bar=True, on_step=False, on_epoch=True)
        return loss.mean()
    
    def sparse_input_training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if isinstance(self.model.config.per_frame_patch_count, list):
            idx = int(len(self.model.config.per_frame_patch_count) * self.global_step / self.trainer.max_steps)
            per_frame_patch_count = self.model.config.per_frame_patch_count[idx]
            self.model.patch_selector.per_frame_patch_count = per_frame_patch_count
        else:
            per_frame_patch_count = self.model.config.per_frame_patch_count
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        x, y = batch
        batch_size, timestep_count, *dims = x.shape
        input_patches = []
        new_patch_logits = None
        losses = []
        accs = []
        optimizer.zero_grad()
        start_idx = randint(0, 1)
        for time_idx in range(timestep_count):
            patchified_x = self.model.patchifier(x)
            batch_size, timestep_count, patch_count, embedding_dim = patchified_x.shape
            new_input_patch = self.model.patch_selector(patchified_x[:, time_idx, :, :], patch_logits=new_patch_logits).view(batch_size, 1, per_frame_patch_count, embedding_dim)
            hidden_acts = torch.cat([x for x in input_patches] + [new_input_patch] + [
                torch.zeros(batch_size, timestep_count-len(input_patches)-1, per_frame_patch_count, embedding_dim, dtype=x.dtype, device=x.device)
            ], dim=1)
            for layer in self.model.transformer_layers:
                hidden_acts = layer(hidden_acts)
            _new_class_logits, _new_patch_logits = self.model.head(hidden_acts)
            new_class_logits = _new_class_logits[:, time_idx, :]
            new_patch_logits = _new_patch_logits[:, time_idx, :]
            if (time_idx + start_idx) % 2 == 0:
                new_loss = 2*nn.functional.cross_entropy(new_class_logits, y)/timestep_count
                self.manual_backward(new_loss)
                new_patch_logits = new_patch_logits.detach()
            with torch.no_grad():
                input_patches.append(new_input_patch.detach())
                losses.append(nn.functional.cross_entropy(new_class_logits, y))
                accs.append(get_accuracy(new_class_logits, y))
        optimizer.step()
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        final_loss = losses[-1]
        avg_acc = sum(accs) / len(accs)
        final_acc = accs[-1]
        self.log('train_loss', avg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_loss_final', final_loss, prog_bar=True, on_step=True)
        self.log('train_acc', avg_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_acc_final', final_acc, prog_bar=True, on_step=True, on_epoch=True)
    
    def random_sequence_prediction(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        patches = []
        losses = []
        accs = []
        patchified_x = self.model.patchifier(x)
        batch_size, timestep_count, patch_count, embedding_dim = patchified_x.shape
        per_frame_patch_count = self.model.patch_selector.per_frame_patch_count
        for time_idx in range(timestep_count):
            D = nn.functional.one_hot(
                torch.from_numpy(np.stack([np.random.choice(patch_count, size=per_frame_patch_count, replace=False) for _ in range(batch_size)])),
                num_classes=patch_count
            )
            patch = (D*patchified_x[:, time_idx, ...].unsqueeze(1)).sum(dim=2)
            patches.append(patch)
            hidden_acts = torch.cat(patches + [torch.zeros(batch_size, timestep_count-len(patches), per_frame_patch_count, embedding_dim, dtype=x.dtype, device=x.device)], dim=1)
            for layer in self.model.transformer_layers:
                hidden_acts = layer(hidden_acts)
            _new_class_logits, _ = self.model.head(hidden_acts)
            new_class_logits = _new_class_logits[:, time_idx, :]
            losses.append(nn.functional.cross_entropy(new_class_logits, y))
            accs.append(get_accuracy(new_class_logits, y))
            avg_loss = sum(losses) / len(losses)
            final_loss = losses[-1]
            avg_acc = sum(accs) / len(accs)
            final_acc = accs[-1]
            self.log('bl_loss', avg_loss, prog_bar=False, on_step=False, on_epoch=True)
            self.log('bl_loss_final', final_loss, prog_bar=True, on_step=True)
            self.log('bl_acc', avg_acc, prog_bar=False, on_step=False, on_epoch=True)
            self.log('bl_acc_final', final_acc, prog_bar=True, on_step=True, on_epoch=True)

    def training_step(self, *args, **kwargs):
        if self.model.config.sparse_inputs:
            return self.sparse_input_training_step(*args, **kwargs)
        else:
            return self.standard_training_step(*args, **kwargs)
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        #self.random_sequence_prediction(batch, batch_idx)
        x, y = batch
        logits = self.model(x)
        batch_size, timesteps, class_count = logits.shape
        logits = logits.reshape(batch_size*timesteps, class_count)
        y = y.reshape(batch_size, 1).expand(-1, timesteps).reshape(batch_size*timesteps)
        loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1, reduction='none').reshape(batch_size, timesteps).mean(dim=0)
        acc = get_accuracy(logits, y, avg_result=False).reshape(batch_size, timesteps).mean(dim=0)
        self.log('val_loss', loss.mean(), prog_bar=False, on_step=True)
        self.log('val_loss_final', loss[-1], prog_bar=True, on_step=True)
        self.log('val_acc', acc.mean(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('val_acc_final', acc[-1], prog_bar=True, on_step=False, on_epoch=True)
        return loss.mean()