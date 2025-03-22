import os

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from lightning.pytorch.callbacks import Callback

class UniformCheckpoint(Callback):
    def __init__(self, save_dir, total_steps, checkpoint_count):
        super().__init__()
        self.save_dir = save_dir
        self.total_steps = total_steps
        self.checkpoint_count = checkpoint_count
        self.checkpoint_steps = np.linspace(0, total_steps, checkpoint_count).astype(int)[1:-1]
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.checkpoint_steps:
            checkpoint_path = os.path.join(self.save_dir, f'step={step}.ckpt')
            trainer.save_checkpoint(checkpoint_path)
    
    def on_train_start(self, trainer, pl_module):
        checkpoint_path = os.path.join(self.save_dir, 'initial_checkpoint.ckpt')
        trainer.save_checkpoint(checkpoint_path)
    
    def on_train_end(self, trainer, pl_module):
        checkpoint_path = os.path.join(self.save_dir, 'final_checkpoint.ckpt')
        trainer.save_checkpoint(checkpoint_path)

@torch.no_grad()
def get_rms_grad(model):
    rms_grad, param_count = 0.0, 0
    for param in model.parameters():
        if param.grad is not None:
            rms_grad += (param.grad**2).sum().item()
            param_count += torch.numel(param)
    rms_grad = (rms_grad / param_count)**0.5
    return rms_grad

def extract_trace(trace):
    x = np.array([u.step for u in trace])
    y = np.array([u.value for u in trace])
    return x, y

def extract_training_curves(save_dir):
    ea = event_accumulator.EventAccumulator(save_dir) #os.path.join(save_dir, 'lightning_output', 'version_0'))
    ea.Reload()
    training_curves = {key: extract_trace(ea.Scalars(key)) for key in ea.Tags()['scalars']}
    np.savez(os.path.join(save_dir, 'training_curves.npz'), **training_curves)

def load_training_curves(save_dir):
    if not os.path.exists(os.path.join(save_dir, 'training_curves.npz')):
        extract_training_curves(save_dir)
    curves_database = np.load(os.path.join(save_dir, 'training_curves.npz'), allow_pickle=True)
    training_curves = {key: curves_database[key] for key in curves_database.files}
    return training_curves

def training_complete(save_dir):
    return all((
        os.path.exists(os.path.join(save_dir, 'training_curves.npz')),
        os.path.exists(os.path.join(save_dir, 'trial_config.yaml'))
    ))