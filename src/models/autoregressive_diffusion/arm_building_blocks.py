from dataclasses import dataclass
from typing import Optional
import os

import torch
from torch import nn

GPT2_DEFAULT_KWARGS = {
    's': {'block_size': 1024, 'vocab_size': 50257, 'layer_count': 12, 'head_count': 12, 'embedding_dim': 768, 'bias': True},
    'm': {'block_size': 1024, 'vocab_size': 50257, 'layer_count': 24, 'head_count': 16, 'embedding_dim': 1024, 'bias': True},
    'l': {'block_size': 1024, 'vocab_size': 50257, 'layer_count': 36, 'head_count': 20, 'embedding_dim': 1280, 'bias': True},
    'xl': {'block_size': 1024, 'vocab_size': 50257, 'layer_count': 48, 'head_count': 25, 'embedding_dim': 1600, 'bias': True}
}

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304
    layer_count: int = 12
    head_count: int = 12
    embedding_dim: int = 768
    dropout_rate: float = 0.0
    bias: bool = False
    pretrained_model_path: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.block_size, int) and (self.block_size > 0)
        assert isinstance(self.vocab_size, int) and (self.vocab_size > 0)
        assert isinstance(self.layer_count, int) and (self.layer_count > 0)
        assert isinstance(self.head_count, int) and (self.head_count > 0)
        assert isinstance(self.embedding_dim, int) and (self.embedding_dim > 0) and (self.embedding_dim % self.head_count == 0)
        assert isinstance(self.dropout_rate, float) and (0 <= self.dropout_rate < 1)
        assert isinstance(self.bias, bool)
        if self.pretrained_model_path is not None:
            assert os.path.exists(self.pretrained_model_path)

class Norm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(self.config.embedding_dim))
        self.bias = nn.Parameter(torch.zeros(self.config.embedding_dim)) if self.config.bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1.e-5)

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=self.config.bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embedding_dim = x.shape
        assert embedding_dim == self.config.embedding_dim
        q, k, v = map(
            lambda x: x.view(batch_size, token_count, embedding_dim//self.config.head_count).transpose(1, 2),   
            self.to_qkv(x).split(embedding_dim, dim=2)
        )
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout_rate if self.training else 0., is_causal=True)
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_dim)
        out = self.dropout(self.to_out(pre_out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dense_1 = nn.Linear(self.config.embedding_dim, 4*self.config.embedding_dim, bias=self.config.bias)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(4*self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.act(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x