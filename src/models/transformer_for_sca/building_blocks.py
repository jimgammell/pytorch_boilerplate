from typing import Optional
from math import sqrt, ceil

import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

from .config import TransformerConfig

class NormLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_num: Optional[int] = None):
        super().__init__()
        self.config = config
        self.block_num = layer_num
        self.weight = nn.Parameter(torch.ones(self.config.embedding_dim))
        if self.config.rescale_norm_outputs:
            assert layer_num is not None
            self.weight.data /= sqrt(layer_num)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.config.norm_eps)

class AttentionLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.head_dim = self.config.embedding_dim // self.config.attn_head_count
        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=self.config.bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.out_dropout = nn.Dropout(self.config.dropout)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        nn.init.normal_(self.to_qkv.weight, mean=0.0, std=sqrt(2./5/self.config.embedding_dim))
        nn.init.normal_(self.to_out.weight, mean=0.0, std=1/sqrt(self.config.embedding_dim)/self.config.layer_count)
        if self.config.bias:
            nn.init.constant_(self.to_qkv.bias, 0)
            nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embedding_dim = x.shape
        qkv = self.to_qkv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(
            lambda x: x.view(batch_size, token_count, self.config.attn_head_count, self.head_dim).transpose(1, 2), qkv
        )
        q, k = map(lambda x: self.rotary_emb.rotate_queries_or_keys(x), (q, k))
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=False, scale=1/sqrt(self.head_dim))
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_dim)
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.w12 = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.bias)
        self.w3 = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        nn.init.normal_(self.w12.weight, mean=0.0, std=sqrt(2./5/self.config.embedding_dim))
        nn.init.normal_(self.w3.weight, mean=0.0, std=1/sqrt(self.config.embedding_dim)/self.config.layer_count)
        if self.config.bias:
            nn.init.constant_(self.w12.bias, 0)
            nn.init.constant_(self.w3.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).split(self.config.embedding_dim, dim=2)
        out = self.w3(nn.functional.silu(x1)*x2)
        out = self.dropout(out)
        return out

class AttentionPoolingLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.head_dim = self.config.embedding_dim // self.config.attn_head_count
        self.pool_queries = nn.Parameter(torch.randn(1, self.config.output_head_count, self.config.embedding_dim))
        self.to_kv = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.output_head_classes, bias=self.config.bias)
        nn.init.normal_(self.to_kv.weight, mean=0.0, std=sqrt(2./5/self.config.embedding_dim))
        nn.init.xavier_uniform_(self.to_out.weight)
        if self.config.bias:
            nn.init.constant_(self.to_kv.bias, 0)
            nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x):
        batch_size, token_count, embedding_dim = x.shape
        q = self.pool_queries.expand(batch_size, -1, -1)
        kv = self.to_kv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(
            lambda x: x.view(batch_size, -1, self.config.attn_head_count, self.head_dim).transpose(1, 2), (q, *kv)
        )
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, self.config.output_head_count, embedding_dim)
        logits = self.to_out(pre_out)
        return logits

class Patchifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv1d(1, self.config.embedding_dim, kernel_size=self.config.patch_size, stride=self.config.patch_size, bias=self.config.bias)
    
    def forward(self, x):
        batch_size, _, dim = x.shape
        padding = self.config.patch_size*ceil(dim/self.config.patch_size) - dim
        x = torch.cat([x, torch.zeros(batch_size, 1, padding, dtype=x.dtype, device=x.device)])
        x = self.patch_embedding(x).transpose(1, 2)
        return x