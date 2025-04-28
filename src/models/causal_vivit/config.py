from typing import Optional, Literal, Union
from dataclasses import dataclass
from math import log2, isfinite
from enum import Enum

class PretrainedModelURLs(Enum):
    DINOv2_S  = r'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
    DINOv2_B  = r'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
    DINOv2_L  = r'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
    DEIT3_S = r'https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth'
    DEIT3_B = r'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth'
    DEIT3_L = r'https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth'
    DEIT_T    = r'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
    DEIT_S    = r'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'
    DEIT_B    = r'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth'

class ViTConf(Enum):
    TINY = dict(
        patch_dim = 16,
        transformer_head_count = 3,
        transformer_layer_count = 12,
        transformer_hidden_dim = 192,
        transformer_mlp_expansion_ratio = 4
    )
    SMALL = dict(
        patch_dim = 16,
        transformer_head_count = 6,
        transformer_layer_count = 12,
        transformer_hidden_dim = 384,
        transformer_mlp_expansion_ratio = 4
    )
    BASE = dict(
        patch_dim = 16,
        transformer_head_count = 12,
        transformer_layer_count = 12,
        transformer_hidden_dim = 768,
        transformer_mlp_expansion_ratio = 4
    )
    LARGE = dict(
        patch_dim = 16,
        transformer_head_count = 16,
        transformer_layer_count = 24,
        tansformer_hidden_dim = 1024,
        transformer_mlp_expansion_ratio = 4
    )

@dataclass
class Config:
    input_channels: int = 3
    input_spatial_dim: int = 256
    out_dim: int = 10
    patch_dim: int = 16
    max_input_temporal_dim: int = 32
    transformer_head_count: int = 12
    transformer_layer_count: int = 12
    transformer_hidden_dim: int = 768
    transformer_mlp_expansion_ratio: int = 4
    dropout: float = 0.0
    include_downsampled_patches: bool = False
    sparse_inputs: bool = False
    per_frame_patch_count: Optional[int] = None
    gumbel_temp: Optional[float] = None
    vit_conf: Optional[Union[str, ViTConf]] = None
    pretrained_model: Optional[Union[str, PretrainedModelURLs]] = None

    def __post_init__(self):
        if self.vit_conf is not None:
            if isinstance(self.vit_conf, str):
                assert self.vit_conf in [x.name for x in ViTConf]
                self.vit_conf = ViTConf[self.vit_conf]
            for key, val in self.vit_conf.value.items():
                setattr(self, key, val)
        if self.pretrained_model is not None:
            if isinstance(self.pretrained_model, str):
                assert self.pretrained_model in [x.name for x in PretrainedModelURLs]
                self.pretrained_model = PretrainedModelURLs[self.pretrained_model]
        assert isinstance(self.input_channels, int) and (self.input_channels > 0)
        assert isinstance(self.input_spatial_dim, int) and (self.input_spatial_dim > 0)
        assert isinstance(self.patch_dim, int) and (self.patch_dim > 0) and (self.input_spatial_dim % self.patch_dim == 0)
        assert isinstance(self.max_input_temporal_dim, int) and (self.max_input_temporal_dim > 0)
        assert isinstance(self.transformer_head_count, int) and (self.transformer_head_count > 0)
        assert isinstance(self.transformer_layer_count, int) and (self.transformer_layer_count > 0)
        assert isinstance(self.transformer_hidden_dim, int) and (self.transformer_hidden_dim > 0) and (self.transformer_hidden_dim % self.transformer_head_count == 0)
        assert isinstance(self.transformer_mlp_expansion_ratio, int) and (self.transformer_mlp_expansion_ratio > 0)
        assert isinstance(self.dropout, float) and (0 <= self.dropout < 1)
        assert isinstance(self.include_downsampled_patches, bool)
        assert isinstance(self.sparse_inputs, bool)
        self.patch_count_without_downsampling = (self.input_spatial_dim//self.patch_dim)**2
        if self.include_downsampled_patches:
            self.image_resolutions = int(log2(self.input_spatial_dim//self.patch_dim)) + 1
        else:
            self.image_resolutions = 1
        self.patch_count = sum((self.input_spatial_dim//(self.patch_dim*2**x))**2 for x in range(self.image_resolutions))
        self.transformer_head_dim = self.transformer_hidden_dim//self.transformer_head_count
        if self.sparse_inputs:
            assert isinstance(self.per_frame_patch_count, int) and (1 <= self.per_frame_patch_count < self.patch_count)
            assert isinstance(self.gumbel_temp, float) and (0. < self.gumbel_temp) and isfinite(self.gumbel_temp)
        else:
            assert self.per_frame_patch_count is None
            assert self.gumbel_temp is None