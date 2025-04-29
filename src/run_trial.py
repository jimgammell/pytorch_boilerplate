import os
import argparse
import yaml
from dataclasses import replace

from common import *
import datasets
import models
from training_modules.supervised_classification import SupervisedClassificationTrainer
from training_modules.sequential_image_classifier import SequentialImageClassifierTrainer
from training_modules.supervised_video import SupervisedVideoTrainer
from utils.flatten_dict import *

def train_supervised_classifier(args, default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    assert args.dataset is not None
    assert args.nn_arch is not None
    trainer = SupervisedClassificationTrainer(
        args.nn_arch, default_model_config_kwargs, default_training_config_kwargs, args.dataset, dataset_kwargs, datamodule_config_kwargs
    )
    trial_name = args.trial_name or f'{args.dataset}_{args.nn_arch}'
    save_dir = os.path.join(OUTPUT_DIR, trial_name)
    os.makedirs(save_dir, exist_ok=True)
    init_logger(print=not(args.quiet), logfile=os.path.join(save_dir, 'log'), level=args.log_level)
    trainer.run(os.path.join(save_dir, 'trainer_output'))

def train_sequential_image_classifier(args, default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    assert args.dataset is not None
    assert args.nn_arch is not None
    trainer = SequentialImageClassifierTrainer(
        args.nn_arch, default_model_config_kwargs, default_training_config_kwargs, args.dataset, dataset_kwargs, datamodule_config_kwargs
    )
    trial_name = args.trial_name or f'{args.dataset}_{args.nn_arch}'
    save_dir = os.path.join(OUTPUT_DIR, trial_name)
    os.makedirs(save_dir, exist_ok=True)
    init_logger(print=not(args.quiet), logfile=os.path.join(save_dir, 'log'), level=args.log_level)
    if args.tune_lr:
        trainer.lr_sweep(os.path.join(save_dir, 'tune_lr'))
    trainer.run(os.path.join(save_dir, 'trainer_output'))

def train_video_discriminative_model(args, nn_arch, dataset, default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    trainer = SupervisedVideoTrainer(
        nn_arch, default_model_config_kwargs, default_training_config_kwargs, dataset, dataset_kwargs, datamodule_config_kwargs
    )
    trial_name = args.trial_name or f'{dataset}_{nn_arch}'
    save_dir = os.path.join(OUTPUT_DIR, trial_name)
    os.makedirs(save_dir, exist_ok=True)
    init_logger(print=not(args.quiet), logfile=os.path.join(save_dir, 'log'), level=args.log_level)
    if args.tune_lr:
        trainer.lr_sweep(os.path.join(save_dir, 'tune_lr'))
    trainer.run(os.path.join(save_dir, 'trainer_output'))

def compute_parametric_stats(args):
    assert args.dataset is not None
    assert False # TODO

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', required=True)
    parser.add_argument(
        '--trial-name', default=None, action='store',
        help=f'Outputs will be stored in `{os.path.join(OUTPUT_DIR, "<TRIAL_NAME>")}.'
    )
    parser.add_argument(
        '--log-level', default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='How much to print/log. Probably best to set this to `DEBUG` in most cases, but you can make the program more-terse if you want.'
    )
    parser.add_argument(
        '--quiet', default=False, action='store_true',
        help=f'Whether or not we should print to the terminal. Things will print to `{os.path.join(OUTPUT_DIR, "<TRIAL_NAME>", "log")} regardless.'
    )
    parametric_stats = subparsers.add_parser('compute-parametric-stats')
    parametric_stats.add_argument(
        '--dataset', action='store', default=None, type=str, choices=[x.value for x in datasets.AVAILABLE_DATASETS],
        help='Which dataset to compute stats for.'
    )
    supervised_classification_parser = subparsers.add_parser('supervised-classify')
    supervised_classification_parser.add_argument(
        '--dataset', action='store', default=None, type=str, choices=[x.value for x in datasets.AVAILABLE_DATASETS],
        help='Which dataset to train on.'
    )
    supervised_classification_parser.add_argument(
        '--nn-arch', action='store', default=None, type=str, choices=[x.value for x in models.AVAILABLE_MODELS],
        help='Which model architecture to use for the ARLM.'
    )
    supervised_classification_parser.add_argument(
        '--config-file', action='store', default=None, choices=AVAILABLE_CONFIG_NAMES,
        help=f'Which hyperparameter config file to use: `{os.path.join(CONFIG_DIR, "<CONFIG_FILE>.yaml")}`. This file must exist and be properly set up.'
    )
    sequential_image_classification_parser = subparsers.add_parser('sequential-image-classify')
    sequential_image_classification_parser.add_argument(
        '--dataset', action='store', default=None, type=str, choices=[x.value for x in datasets.AVAILABLE_DATASETS],
        help='Which dataset to train on.'
    )
    sequential_image_classification_parser.add_argument(
        '--nn-arch', action='store', default=None, type=str, choices=[x.value for x in models.AVAILABLE_MODELS],
        help='Which model architecture to use.'
    )
    sequential_image_classification_parser.add_argument(
        '--config-file', action='store', default=None, choices=AVAILABLE_CONFIG_NAMES,
        help=f'Which hyperparameter config file to use: `{os.path.join(CONFIG_DIR, "<CONFIG_FILE>.yaml")}`. This file must exist and be properly set up.'
    )
    sequential_image_classification_parser.add_argument(
        '--tune-lr', action='store_true', default=False, help='Whether or not to run a learning rate sweep before training.'
    )
    supervised_video_parser = subparsers.add_parser('supervised-video')
    supervised_video_parser.add_argument(
        '--config-file', action='store', default=None, choices=AVAILABLE_CONFIG_NAMES,
        help=f'Which hyperparameter config file to use: `{os.path.join(CONFIG_DIR, "<CONFIG_FILE>.yaml")}`. This file must exist and be properly set up.'
    )
    supervised_video_parser.add_argument(
        '--tune-lr', action='store_true', default=False, help='Whether or not to run a learning rate sweep before training.'
    )
    supervised_video_parser.add_argument(
        '--override-config', action='store', nargs='*', default=[], help='Override one of the configuration settings for this trial.'
    )
    args = parser.parse_args()

    if args.action in ['supervised-classify']:
        config_name = args.config_file or f'{args.dataset}_{args.nn_arch}'
        config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_model_config_kwargs = config['default_model_config']
        default_training_config_kwargs = config['default_training_config']
        datamodule_config_kwargs = config['datamodule_config']
        dataset_kwargs = config['dataset_config']
        if args.action == 'supervised-classify':
            train_supervised_classifier(args, default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs)
    elif args.action in ['sequential-image-classify']:
        config_name = args.config_file or f'{args.dataset}_{args.nn_arch}'
        config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_model_config_kwargs = config['default_model_config']
        default_training_config_kwargs = config['default_training_config']
        datamodule_config_kwargs = config['datamodule_config']
        dataset_kwargs = config['dataset_config']
        if args.action == 'sequential-image-classify':
            train_sequential_image_classifier(args, default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs)
    elif args.action == 'supervised-video':
        config_name = args.config_file or f'{args.dataset}_{args.nn_arch}'
        config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = flatten_dict(config)
        for setting in args.override_config:
            key, val = setting.split('=')
            config['.'+key] = yaml.safe_load(val)
        config = unflatten_dict(config)['']
        default_model_config_kwargs = config['default_model_config']
        default_training_config_kwargs = config['default_training_config']
        datamodule_config_kwargs = config['datamodule_config']
        dataset_kwargs = config['dataset_config']
        timesteps = default_model_config_kwargs['max_input_temporal_dim']
        datamodule_config_kwargs['timestep_count'] = timesteps
        dataset_kwargs['timesteps'] = timesteps
        train_video_discriminative_model(args, config['nn_arch'], config['dataset'], default_training_config_kwargs, default_model_config_kwargs, datamodule_config_kwargs, dataset_kwargs)
    elif args.action  == 'compute-parametric-stats':
        compute_parametric_stats(args)
    else:
        assert False

if __name__ == '__main__':
    main()