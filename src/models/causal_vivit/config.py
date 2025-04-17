from dataclasses import dataclass

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

        self.patch_count = (self.input_spatial_dim//self.patch_dim)**2
        self.transformer_head_dim = self.transformer_hidden_dim//self.transformer_head_count