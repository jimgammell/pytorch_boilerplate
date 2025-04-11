from typing import Dict, Any
from copy import copy

import yaml
from lightning import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from common import *
from ..utils import *
import datasets
from datasets.datamodule import DataModule, DataModuleConfig
from .module import SequentialImageClassifierModule
from .config import Config
from .plot_things import *

class SequentialImageClassifierTrainer:
    def __init__(self,
        classifier_name: str,
        base_classifier_config_kwargs: Dict[str, Any],
        base_training_config_kwargs: Dict[str, Any],
        dataset_name: str,
        dataset_kwargs: Dict[str, Any],
        datamodule_config_kwargs: Dict[str, Any]
    ):
        self.classifier_name = classifier_name
        self.base_classifier_config_kwargs = base_classifier_config_kwargs
        self.base_training_config_kwargs = base_training_config_kwargs
        self.dataset_name = dataset_name
        self.dataset_kwargs = dataset_kwargs
        self.datamodule_config_kwargs = datamodule_config_kwargs

        self.train_dataset, self.test_dataset = datasets.load(self.dataset_name, **self.dataset_kwargs)
        self.datamodule = DataModule(
            base_train_dataset=self.train_dataset, config=DataModuleConfig(**self.datamodule_config_kwargs), test_dataset=self.test_dataset
        )

    def run(self,
        save_dir: str,
        classifier_config_kwargs: Dict[str, Any] = {},
        training_config_kwargs: Dict[str, Any] = {},
        may_resume: bool = True
    ):
        _classifier_config_kwargs = copy(self.base_classifier_config_kwargs)
        _classifier_config_kwargs.update(classifier_config_kwargs)
        classifier_config_kwargs = _classifier_config_kwargs
        _training_config_kwargs = copy(self.base_training_config_kwargs)
        _training_config_kwargs.update(training_config_kwargs)
        training_config_kwargs = _training_config_kwargs
        can_run = True
        checkpoint_path = None
        if os.path.exists(save_dir):
            if len(os.listdir(save_dir)) == 0:
                pass
            old_config_path = os.path.join(save_dir, 'trial_config.yaml')
            if not os.path.exists(old_config_path):
                logger.error(f'There is a non-empty directory at {save_dir} which is not a valid trial. Skipping training.')
                can_run = False
            else:
                with open(old_config_path, 'r') as f:
                    old_config = yaml.load(f, Loader=yaml.FullLoader)
                if (old_config['classifier_config'] != classifier_config_kwargs) or (old_config['training_config'] != training_config_kwargs):
                    logger.error(f'There is a trial in {save_dir} which does not have the same hyperparameters as the current trial. Skipping training.')
                    can_run = False
                elif training_complete(save_dir):
                    logger.info(f'There is a complete trial in {save_dir} with the same hyperparameters as the current trial. Skipping training.')
                    can_run = True
                elif not may_resume:
                    logger.error(f'There is a partially-complete trial in {save_dir} with the same hyperparameters as the current trial. Skipping training because may_resume=False.')
                    can_run = False
                else:
                    checkpoint_steps = [int(f.split('=')[-1].split('.')[0]) for f in os.listdir(save_dir) if f.split('=')[0] == 'step' and f.split('.')[-1] == 'ckpt']
                    if os.path.exists(os.path.join(save_dir, 'final_checkpoint.ckpt')):
                        checkpoint_path = os.path.join(save_dir, 'final_checkpoint.ckpt')
                    elif len(checkpoint_steps) > 0:
                        checkpoint_path = os.path.join(save_dir, f'step={max(checkpoint_steps)}.ckpt')
                    elif os.path.exists(os.path.join(save_dir, 'initial_checkpoint.ckpt')):
                        checkpoint_path = os.path.join(save_dir, 'initial_checkpoint.ckpt')
                    if checkpoint_path is not None:
                        logger.info(f'There is a partially-complete trial in {save_dir} with the same hyperparameters as the current trial. Resuming training from {checkpoint_path}.')
        if can_run:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'trial_config.yaml'), 'w') as f:
                yaml.dump({'classifier_config': classifier_config_kwargs, 'training_config': training_config_kwargs}, f, default_flow_style=False)
            training_config = Config(**training_config_kwargs)
            if not training_complete(save_dir):
                training_module = SequentialImageClassifierModule(
                    self.classifier_name, classifier_config_kwargs, training_config
                )
                early_stopping_checkpoint = ModelCheckpoint(monitor='val_rank', mode='min', save_top_k=1, dirpath=save_dir, filename='best_checkpoint')
                progress_bar = TQDMProgressBar(refresh_rate=1) #10)
                trainer = LightningTrainer(
                    max_steps=training_config.training_steps,
                    #log_every_n_steps=100,
                    precision=training_config.dtype,
                    logger=TensorBoardLogger(save_dir, name='', version=''),
                    callbacks=[]#[early_stopping_checkpoint, progress_bar]
                )
                trainer.fit(training_module, datamodule=self.datamodule, ckpt_path=checkpoint_path)
            else:
                print('Loading trained model.')
                checkpoint_filename = [x for x in os.listdir(os.path.join(save_dir, 'checkpoints')) if x.split('.')[-1] == 'ckpt'][0]
                training_module = SequentialImageClassifierModule.load_from_checkpoint(os.path.join(save_dir, 'checkpoints', checkpoint_filename))
                trainer = LightningTrainer(
                    max_steps=training_config.training_steps,
                    #log_every_n_steps=100,
                    precision=training_config.dtype,
                    logger=TensorBoardLogger(save_dir, name='', version=''),
                    callbacks=[]#[early_stopping_checkpoint, progress_bar]
                )
            test_results = trainer.test(training_module, datamodule=self.datamodule, verbose=False)
            logger.info(f'Test results: {test_results}')
            extract_training_curves(save_dir)
        if os.path.exists(os.path.join(save_dir, 'training_curves.npz')):
            training_curves = load_training_curves(save_dir)
            plot_training_curves(training_curves, save_dir)