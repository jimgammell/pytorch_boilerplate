import numpy as np
import torch
from torch import nn

from ..base_module import BaseModule
from .config import Config

class PatchEmbedder(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
    
    def construct(self):
        self.to_patches = nn.Conv2d(
            self.config.input_channel_count, self.config.embedding_dim, kernel_size=self.config.patch_size, stride=self.config.patch_size, bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        x = self.to_patches(x).reshape(batch_size, self.config.embedding_dim, self.config.patch_count).permute(0, 2, 1)
        return x

class Norm(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
    
    def construct(self):
        self.norm = nn.LayerNorm(self.config.embedding_dim, eps=1e-6, elementwise_affine=True, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class Attention(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()

    def construct(self):
        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=self.config.bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.out_dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embedding_dim = x.shape
        assert embedding_dim == self.config.embedding_dim
        qkv = self.to_qkv(x).reshape(batch_size, token_count, 3, self.config.attn_head_count, self.config.attn_head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout).permute(0, 2, 1, 3).reshape(batch_size, token_count, embedding_dim)
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        return out

class FeedForward(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
    
    def construct(self):
        self.fc1 = nn.Linear(self.config.embedding_dim, self.config.mlp_dim, bias=self.config.bias)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.mlp_dim, self.config.embedding_dim, bias=self.config.bias)
        self.dropout2 = nn.Dropout(self.config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class AttentionPool(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
        self.init_weights()
    
    def construct(self):
        self.q = nn.Parameter(torch.zeros(1, 1, self.config.embedding_dim))
        self.to_kv = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.out_dropout = nn.Dropout(self.config.dropout)
        self.norm1 = Norm(self.config)
        self.norm2 = Norm(self.config)
        self.fnn = FeedForward(self.config)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.q, std=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embedding_dim = x.shape
        x = self.norm1(x)
        q = self.q.expand(batch_size, -1, -1).reshape(batch_size, self.config.attn_head_count, 1, self.config.attn_head_dim)
        kv = self.to_kv(x).reshape(batch_size, token_count, 2, self.config.attn_head_count, self.config.attn_head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(batch_size, 1, embedding_dim)
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        out = out + self.fnn(self.norm2(out))
        return out