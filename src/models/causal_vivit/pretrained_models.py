from typing import Union, Dict, Any
from enum import Enum

import torch
from torch import nn

from common import *
from utils.download_utils import download
from ..base_module import BaseModule
from .config import Config, ViTConf, PretrainedModelURLs

def _load_weight(dest: torch.Tensor, src: torch.Tensor):
    assert dest.shape == src.shape
    dest.data = src

def _check_config_args(config: Config, args: Dict[str, Any]):
    for key, val in args.items():
        assert hasattr(config, key)
        assert getattr(config, key) == val

def load_pretrained_weights(model: BaseModule, pretrained_model: Union[str, PretrainedModelURLs]):
    if isinstance(pretrained_model, str):
        assert any(pretrained_model == x.name for x in PretrainedModelURLs)
        pretrained_model = PretrainedModelURLs(pretrained_model)
    name = pretrained_model.name
    url = pretrained_model.value
    vit_size = name.split('_')[-1]
    if vit_size == 'T':
        _check_config_args(model.config, ViTConf.TINY.value)
    elif vit_size == 'S':
        _check_config_args(model.config, ViTConf.SMALL.value)
    elif vit_size == 'B':
        _check_config_args(model.config, ViTConf.BASE.value)
    elif vit_size == 'L':
        _check_config_args(model.config, ViTConf.LARGE.value)
    else:
        assert False
    pretrained_weights_dir = os.path.join(RESOURCE_DIR, 'pretrained_weights')
    pretrained_weights_path = os.path.join(pretrained_weights_dir, url.split('/')[-1])
    download(url, pretrained_weights_path, verbose=True)
    pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')['model']
    for patch_embedder in model.patchifier.patch_embedders:
        _load_weight(patch_embedder.weight, pretrained_weights['patch_embed.proj.weight'])
        _load_weight(patch_embedder.bias, pretrained_weights['patch_embed.proj.bias'])
    _load_weight(
        model.patchifier.spatial_position_embedding[:, :, :model.config.patch_count_without_downsampling, :],
        nn.functional.interpolate(
            pretrained_weights['pos_embed'].permute(0, 2, 1), size=model.config.patch_count_without_downsampling, mode='linear'
        ).permute(0, 2, 1).contiguous().view(1, 1, model.config.patch_count_without_downsampling, model.config.transformer_hidden_dim)
    )
    for layer_idx, layer in enumerate(model.transformer_layers):
        if hasattr(layer, 'temporal_attention'):
            nn.init.constant_(layer.temporal_attention.to_out.weight, 0)
            nn.init.constant_(layer.temporal_attention.to_out.bias, 0)
        if hasattr(layer, 'spatial_attention'):
            _load_weight(layer.spatial_attention.to_qkv.weight, pretrained_weights[f'blocks.{layer_idx}.attn.qkv.weight'])
            _load_weight(layer.spatial_attention.to_qkv.bias, pretrained_weights[f'blocks.{layer_idx}.attn.qkv.bias'])
            _load_weight(layer.spatial_attention.to_out.weight, pretrained_weights[f'blocks.{layer_idx}.attn.proj.weight'])
            _load_weight(layer.spatial_attention.to_out.bias, pretrained_weights[f'blocks.{layer_idx}.attn.proj.bias'])
            _load_weight(layer.pre_spatial_attn_norm.norm_layer.weight, pretrained_weights[f'blocks.{layer_idx}.norm1.weight'])
            _load_weight(layer.pre_spatial_attn_norm.norm_layer.bias, pretrained_weights[f'blocks.{layer_idx}.norm1.bias'])
        if hasattr(layer, 'attn'):
            _load_weight(layer.attn.to_qkv.weight, pretrained_weights[f'blocks.{layer_idx}.attn.qkv.weight'])
            _load_weight(layer.attn.to_qkv.bias, pretrained_weights[f'blocks.{layer_idx}.attn.qkv.bias'])
            _load_weight(layer.attn.to_out.weight, pretrained_weights[f'blocks.{layer_idx}.attn.proj.weight'])
            _load_weight(layer.attn.to_out.bias, pretrained_weights[f'blocks.{layer_idx}.attn.proj.bias'])
            _load_weight(layer.pre_attn_norm.norm_layer.weight, pretrained_weights[f'blocks.{layer_idx}.norm1.weight'])
            _load_weight(layer.pre_attn_norm.norm_layer.bias, pretrained_weights[f'blocks.{layer_idx}.norm1.bias'])
        if hasattr(layer, 'fnn'):
            _load_weight(layer.fnn.fc1.weight, pretrained_weights[f'blocks.{layer_idx}.mlp.fc1.weight'])
            _load_weight(layer.fnn.fc1.bias, pretrained_weights[f'blocks.{layer_idx}.mlp.fc1.bias'])
            _load_weight(layer.fnn.fc2.weight, pretrained_weights[f'blocks.{layer_idx}.mlp.fc2.weight'])
            _load_weight(layer.fnn.fc2.bias, pretrained_weights[f'blocks.{layer_idx}.mlp.fc2.bias'])
            _load_weight(layer.pre_fnn_norm.norm_layer.weight, pretrained_weights[f'blocks.{layer_idx}.norm2.weight'])
            _load_weight(layer.pre_fnn_norm.norm_layer.bias, pretrained_weights[f'blocks.{layer_idx}.norm2.bias'])