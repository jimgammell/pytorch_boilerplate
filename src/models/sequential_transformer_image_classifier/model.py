from typing import Optional

import torch
from torch import nn

from ..base_module import BaseModule
from .config import Config
from .building_blocks import *

class TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pre_attn_norm = NormLayer(self.config)
        self.attn = AttentionLayer(self.config)
        self.pre_fnn_norm = NormLayer(self.config)
        self.fnn = FeedForwardLayer(self.config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_norm(x), mask)
        x = x + self.fnn(self.pre_fnn_norm(x))
        return x

class Head(nn.Module):
    def __init__(self, config: Config, next_patch_logits_bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.pre_attn_norm = NormLayer(self.config)
        self.attn = AttentionPoolingLayer(self.config, next_patch_logits_bias=next_patch_logits_bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(self.pre_attn_norm(x), mask)
        return x

class Transformer(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_extractor = PatchExtractor(self.config)
        self.patch_embedder = PatchEmbedder(self.config)
        self.patch_selector = PatchSelector(self.config)
        self.transformer_layers = nn.ModuleList(TransformerLayer(self.config) for _ in range(self.config.transformer_layer_count))
        self.head = Head(self.config, next_patch_logits_bias=self.patch_selector.prior_pos_logits)
    
    def get_embedded_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(x)
        embedded_patches = self.patch_embedder(patches)
        return embedded_patches
    
    def next_iter(self, embedded_patches: torch.Tensor, seq_indices: torch.Tensor, seq_lengths: torch.Tensor, pos_logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, *_ = embedded_patches.shape
        x, mask = self.patch_selector(embedded_patches, seq_indices, seq_lengths, pos_logits)
        x = self.transformer_layers(x, mask)
        class_logits, next_pos_logits = self.head(x, mask)
        seq_indices[torch.arange(batch_size, device=embedded_patches.device), seq_lengths, :] = torch.multinomial(nn.functional.softmax(next_pos_logits.detach(), dim=-1), 1)
        seq_lengths = seq_lengths + 1
        return class_logits, next_pos_logits, seq_indices, seq_lengths
    
    @torch.no_grad()
    def run_inference(self, x: torch.Tensor, max_iters: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, *_ = x.shape
        if max_iters is None:
            max_iters = self.config.max_sequence_length
        embedded_patches = self.get_embedded_patches(x)
        seq_indices = torch.arange(max_iters, device=x.device).unsqueeze(0).expand(batch_size, -1)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        pos_logits = None
        next_pos_logits = torch.full((batch_size, max_iters, self.config.patch_count), torch.nan, device=x.device)
        class_logits = torch.full((batch_size, max_iters, self.config.output_dim), torch.nan, device=x.device)
        for idx in range(max_iters):
            class_logits[:, idx, :], next_pos_logits[:, idx, :], seq_indices, seq_lengths = self.next_iter(embedded_patches, seq_indices, seq_lengths, pos_logits)
        assert next_pos_logits.isfinite().all()
        assert class_logits.isfinite().all()
        return class_logits, next_pos_logits
    
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        embedded_patches = self.get_embedded_patches(x)
        batch_size, patch_count, patch_dim = embedded_patches.shape
        seq_indices = torch.rand(batch_size, patch_count, device=x.device).argsort(dim=-1)
        seq_lengths = ((patch_count-2)*(torch.rand(batch_size, device=x.device)**2)).to(torch.long)
        pos_logits = torch.zeros(batch_size, patch_count, device=x.device)
        for idx in range(batch_size):
            pos_logits[idx, seq_indices[idx, :seq_lengths[idx]]] = -torch.inf
        _, next_pos_logits, seq_indices, seq_lengths = self.next_iter(embedded_patches, seq_indices, seq_lengths, pos_logits)
        class_logits, _, _, _ = self.next_iter(embedded_patches, seq_indices, seq_lengths, next_pos_logits)
        return class_logits