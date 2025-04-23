from typing import Optional, Literal
from dataclasses import dataclass
from math import log2, isfinite

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
    init_from_deit: Optional[Literal['tiny', 'tiny-distilled', 'small', 'small-distilled', 'base-mae']] = None
    include_downsampled_patches: bool = False
    sparse_inputs: bool = False
    per_frame_patch_count: Optional[int] = None
    gumbel_estimator: Optional[Literal['hard', 'soft']] = None
    gumbel_temp: Optional[float] = None

    def __post_init__(self):
        assert isinstance(self.input_channels, int) and (self.input_channels > 0)
        assert isinstance(self.input_spatial_dim, int) and (self.input_spatial_dim > 0)
        assert isinstance(self.patch_dim, int) and (self.patch_dim > 0) and (self.input_spatial_dim % self.patch_dim == 0)
        assert isinstance(self.max_input_temporal_dim, int) and (self.max_input_temporal_dim > 0)
        assert isinstance(self.transformer_head_count, int) and (self.transformer_head_count > 0)
        assert isinstance(self.transformer_layer_count, int) and (self.transformer_layer_count > 0)
        assert isinstance(self.transformer_hidden_dim, int) and (self.transformer_hidden_dim > 0) and (self.transformer_hidden_dim % self.transformer_head_count == 0)
        assert isinstance(self.transformer_mlp_expansion_ratio, int) and (self.transformer_mlp_expansion_ratio > 0)
        assert isinstance(self.dropout, float) and (0 <= self.dropout < 1)
        assert (self.init_from_deit is None) or (self.init_from_deit in ['tiny', 'tiny-distilled', 'small', 'small-distilled', 'base-mae'])
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
            assert self.gumbel_estimator in ['hard', 'soft']
            assert isinstance(self.gumbel_temp, float) and (0. < self.gumbel_temp) and isfinite(self.gumbel_temp)
        else:
            assert self.per_frame_patch_count is None
            assert self.gumbel_estimator is None
            assert self.gumbel_temp is None