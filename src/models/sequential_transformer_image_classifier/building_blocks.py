from typing import Optional, Tuple
from math import sqrt

import numpy as np
import torch
from torch import nn
from torchvision.ops import MLP

from ..base_module import BaseModule
from .config import Config

class ConvPatchExtractorAndEmbedder(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channel_count, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(384, self.config.embedding_dim, kernel_size=1, stride=1, padding=0)
        )
        self.patch_count = 16**2 + 8**2 + 4**2 + 2**2 + 1**2
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_count, self.config.embedding_dim))
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = []
        x = self.stem(x)
        patches.append(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        patches.append(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        patches.append(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        patches.append(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        patches.append(x)
        patches = torch.cat([patch.flatten(start_dim=2).permute(0, 2, 1) for patch in patches], dim=1)
        patches = patches + self.position_embeddings
        return patches

class PatchExtractor(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.per_res_patch_indices = np.cumsum([0] + self.config.per_res_patch_counts.tolist())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, width, height = x.shape
        assert self.config.input_channel_count == channel_count
        assert self.config.input_dim == width == height
        patches = torch.full(
            (batch_size, self.config.patch_dim, self.config.patch_count),
            torch.nan, device=x.device, dtype=x.dtype
        )
        for idx, (start_patch_idx, end_patch_idx) in enumerate(zip(self.per_res_patch_indices[:-1], self.per_res_patch_indices[1:])):
            patches[:, :, start_patch_idx:end_patch_idx] = nn.functional.unfold(x, kernel_size=self.config.base_patch_dim, stride=self.config.base_patch_dim)
            if idx < self.config.resolutions:
                x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        patches = patches.permute(0, 2, 1)
        return patches

class PatchEmbedder(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.learned_position_embeddings = nn.Parameter(torch.zeros(1, self.config.patch_count, self.config.embedding_dim))
        self.patch_embedder = nn.Linear(self.config.input_channel_count*self.config.base_patch_dim**2, self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        nn.init.xavier_uniform_(self.patch_embedder.weight)
        if self.config.bias:
            nn.init.constant_(self.patch_embedder.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, patch_dim = x.shape
        assert patch_count == self.config.patch_count
        assert patch_dim == self.config.patch_dim
        x = self.patch_embedder(x) + self.learned_position_embeddings
        x = self.dropout(x)
        return x

class PatchSelector(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.prior_pos_logits = nn.Parameter(torch.zeros(1, self.config.patch_count))

    def forward(self,
        x: torch.Tensor, # the full sequence of embedded patches
        seq_indices: torch.Tensor, # the patch order of each of the sequences
        seq_lengths: torch.Tensor, # the lengths of each of the sequences in the batch
        pos_logits: Optional[torch.Tensor] = None # the distribution over the next patch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, patch_count, embedding_dim = x.shape
        assert patch_count == self.config.patch_count
        assert embedding_dim == self.config.embedding_dim
        assert batch_size == seq_indices.size(0) == seq_lengths.size(0)
        assert seq_indices.size(-1) == self.config.max_sequence_length
        # assert (0 <= seq_lengths < self.config.max_sequence_length).all() # hopefully true, but commenting out to avoid CPU-GPU sync
        if pos_logits is None:
            pos_logits = self.prior_pos_logits.expand(batch_size, -1)
        else: # If the sequence length is zero, we ignore input logits and use the prior. Somewhat-efficient way to let us do this for only some elements of a batch.
            zero_mask = (seq_lengths == 0).unsqueeze(1).float()
            pos_logits = zero_mask*self.prior_pos_logits.expand(batch_size, -1) + (1-zero_mask)*pos_logits
        pos_dist = nn.functional.gumbel_softmax(pos_logits, tau=self.config.gumbel_tau, hard=not self.training, dim=-1).unsqueeze(-1)
        output_sequence = torch.gather(x, dim=1, index=seq_indices.unsqueeze(-1).expand(-1, -1, embedding_dim))
        output_sequence[torch.arange(batch_size, device=seq_lengths.device), seq_lengths-1, :] = (x*pos_dist).sum(dim=1)
        attn_mask = torch.arange(self.config.max_sequence_length, device=x.device).unsqueeze(0).expand(batch_size, -1) <= seq_lengths.unsqueeze(1).expand(-1, self.config.max_sequence_length)
        return output_sequence, attn_mask, pos_dist

class NormLayer(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(self.config.embedding_dim, eps=self.config.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, embedding_dim = x.shape
        assert patch_count == self.config.max_sequence_length
        assert embedding_dim == self.config.embedding_dim
        return self.norm(x)

class AttentionLayer(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=False)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        if self.config.bias:
            nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, patch_count, embedding_dim = x.shape
        assert batch_size == mask.size(0)
        assert patch_count == mask.size(1) == self.config.max_sequence_length
        assert embedding_dim == self.config.embedding_dim
        qkv = self.to_qkv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(lambda u: u.view(batch_size, patch_count, self.config.attn_head_count, self.config.attn_head_dim).transpose(1, 2), qkv)
        if mask is not None:
            mask = mask.reshape(batch_size, 1, 1, patch_count).expand(-1, self.config.attn_head_count, patch_count, -1)
        pre_out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.config.dropout if self.training else 0, is_causal=False, scale=1/sqrt(self.config.attn_head_dim)
        )
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, patch_count, embedding_dim)
        out = self.to_out(pre_out)
        out = self.dropout(out)
        return out

r"""class FeedForwardLayer(BaseModule):
    def __init__(self, config: Config, out_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.out_dim = out_dim
        self.w12 = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.bias)
        self.w3 = nn.Linear(self.config.embedding_dim, self.out_dim or self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.config.bias:
            nn.init.constant_(self.w12.bias, 0)
            nn.init.constant_(self.w3.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).split(self.config.embedding_dim, dim=2)
        out = self.w3(nn.functional.silu(x1)*x2)
        out = self.dropout(out)
        return out"""

class FeedForwardLayer(BaseModule):
    def __init__(self, config: Config, out_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.out_dim = out_dim or self.config.embedding_dim
        self.mlp = MLP(
            in_channels=self.config.embedding_dim, hidden_channels=[self.config.mlp_dim, self.out_dim],
            activation_layer=nn.GELU, dropout=self.config.dropout, inplace=None
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.normal_(mod.bias, std=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class AttentionPoolingLayer(BaseModule):
    def __init__(self, config: Config, next_patch_logits_bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.pool_queries = nn.Parameter(torch.zeros(1, 2, self.config.embedding_dim)) # 1 for classification, 1 for next patch prediction
        self.to_kv = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=False)
        self.to_classification_logits = nn.Linear(self.config.embedding_dim, self.config.output_dim, bias=True)
        self.to_next_patch_logits = nn.Linear(self.config.embedding_dim, self.config.patch_count, bias=False)
        self.next_patch_logits_bias = next_patch_logits_bias if next_patch_logits_bias is not None else  nn.Parameter(torch.zeros(1, self.config.patch_count))
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_classification_logits.weight)
        nn.init.xavier_uniform_(self.to_next_patch_logits.weight)
        nn.init.constant_(self.to_classification_logits.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, patch_count, embedding_dim = x.shape
        q = self.pool_queries.expand(batch_size, -1, -1)
        kv = self.to_kv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(lambda x: x.view(batch_size, -1, self.config.attn_head_count, self.config.attn_head_dim).transpose(1, 2), (q, *kv))
        if mask is not None:
            mask = mask.reshape(batch_size, 1, 1, patch_count).expand(-1, self.config.attn_head_count, 2, -1)
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0, is_causal=False)
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, 2, embedding_dim)
        classification_logits = self.to_classification_logits(pre_out[:, 0, :])
        next_patch_logits = self.to_next_patch_logits(pre_out[:, 1, :]) + self.next_patch_logits_bias
        return classification_logits, next_patch_logits