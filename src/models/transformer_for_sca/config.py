from dataclasses import dataclass

@dataclass
class TransformerConfig:
    patch_size: int = 100
    layer_count: int = 12
    attn_head_count: int = 12
    embedding_dim: int = 768
    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-5
    rescale_norm_outputs: bool = True
    output_head_count: int = 16
    output_head_classes: int = 256