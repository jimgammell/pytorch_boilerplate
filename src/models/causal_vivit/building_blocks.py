from typing import Literal, Optional

import torch
from torch import nn
from xformers import ops as xformers_ops
from rotary_embedding_torch import RotaryEmbedding

from ..base_module import BaseModule
from .config import Config

class Patchifier(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.patch_embedders = nn.ModuleList([
            nn.Conv2d(self.config.input_channels, self.config.transformer_hidden_dim, kernel_size=self.config.patch_dim, stride=self.config.patch_dim)
            for _ in range(self.config.image_resolutions)
        ])

        self.dropout = nn.Dropout(self.config.dropout)
        self.spatial_position_embedding = nn.Parameter(torch.empty((1, 1, self.config.patch_count, self.config.transformer_hidden_dim), dtype=torch.float))
    
    def init_weights(self):
        for patch_embedder in self.patch_embedders:
            nn.init.trunc_normal_(patch_embedder.weight, mean=0., std=0.02)
            nn.init.constant_(patch_embedder.bias, 0)
        nn.init.normal_(self.spatial_position_embedding, mean=0., std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape
        assert timesteps == self.config.max_input_temporal_dim
        assert channels == self.config.input_channels
        assert height == width == self.config.input_spatial_dim
        x = x.view(batch_size*timesteps, channels, height, width)
        embedded_patches = torch.zeros(batch_size, timesteps, self.config.patch_count, self.config.transformer_hidden_dim, dtype=x.dtype, device=x.device)
        idx = 0
        for res_idx in range(self.config.image_resolutions):
            embedded_x = self.patch_embedders[res_idx](x).view(batch_size, timesteps, self.config.transformer_hidden_dim, -1).permute(0, 1, 3, 2).contiguous()
            patch_count = embedded_x.size(2)
            embedded_patches[:, :, idx:idx+patch_count, :] = embedded_patches[:, :, idx:idx+patch_count, :] + embedded_x
            if res_idx < self.config.image_resolutions-1:
                x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
                idx += patch_count
        embedded_patches = embedded_patches + self.spatial_position_embedding
        embedded_patches = self.dropout(embedded_patches)
        return embedded_patches

class PatchSelector(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.prior_logits = nn.Parameter(torch.empty((self.config.patch_count,), dtype=torch.float))
    
    def init_weights(self):
        nn.init.trunc_normal_(self.prior_logits, mean=0., std=0.02)
    
    def forward(self, patchified_frame: torch.Tensor, patch_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, patch_count, embedding_dim = patchified_frame.shape
        if patch_logits is None:
            patch_logits = self.prior_logits.reshape(1, patch_count).expand(batch_size, -1)
        with torch.autocast(enabled=False, device_type='cuda'): # 32-bit precision isn't enough for this
            u = torch.rand(patch_logits.shape, device=patch_logits.device, dtype=torch.float64).clamp(1e-12, 1.-1e-12)
            gumbel_noise = -torch.log(-torch.log(u))
            gumbel_noise = gumbel_noise.to(patch_logits.dtype)
        soft_sample = torch.softmax((patch_logits + gumbel_noise)/self.config.gumbel_temp, dim=-1)
        if self.training:
            if self.config.gumbel_estimator == 'soft':
                assert self.config.per_frame_patch_count == 1
                dist = soft_sample
            elif self.config.gumbel_estimator == 'hard':
                with torch.no_grad():
                    idx = soft_sample.topk(self.config.per_frame_patch_count, dim=-1).indices
                    hard_sample = torch.zeros_like(soft_sample).scatter_(-1, idx, 1.0)
                    print([idx.shape, hard_sample.shape])
                dist = hard_sample + soft_sample - soft_sample.detach()
            else:
                assert False
        else:
            idx = soft_sample.topk(self.config.per_frame_patch_count, dim=-1).indices
            dist = torch.zeros_like(soft_sample).scatter_(-1, idx, 1.0)
        print([dist.shape, patchified_frame.shape])
        dist = dist.reshape(batch_size, patch_count, 1).expand(-1, -1, embedding_dim)
        patch = (dist*patchified_frame).sum(dim=1)
        return patch

class NormLayer(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.norm_layer = nn.LayerNorm(self.config.transformer_hidden_dim, eps=1e-6, elementwise_affine=True, bias=True)
    
    def init_weights(self):
        nn.init.constant_(self.norm_layer.weight, 1)
        nn.init.constant_(self.norm_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_layer(x)

class FeedForward(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.fc1 = nn.Linear(self.config.transformer_hidden_dim, self.config.transformer_hidden_dim*self.config.transformer_mlp_expansion_ratio)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.transformer_mlp_expansion_ratio*self.config.transformer_hidden_dim, self.config.transformer_hidden_dim)
        self.dropout2 = nn.Dropout(self.config.dropout)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, mean=0., std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.trunc_normal_(self.fc2.weight, mean=0., std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Attention(BaseModule):
    def __init__(self, config: Config, mode: Literal['spatial', 'temporal'] = 'spatial'):
        self.config = config
        self.mode = mode
        super().__init__()
    
    def construct(self):
        self.to_qkv = nn.Linear(self.config.transformer_hidden_dim, 3*self.config.transformer_hidden_dim)
        self.to_out = nn.Linear(self.config.transformer_hidden_dim, self.config.transformer_hidden_dim)
        self.out_dropout = nn.Dropout(self.config.dropout)
        if self.mode == 'temporal':
            self.rope = RotaryEmbedding(dim=self.config.transformer_head_dim)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.to_qkv.weight, mean=0., std=0.02)
        nn.init.constant_(self.to_qkv.bias, 0)
        nn.init.trunc_normal_(self.to_out.weight, mean=0., std=0.02)
        nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, timesteps, tokens_per_timestep, embedding_dim = x.shape
        assert timesteps == self.config.max_input_temporal_dim
        assert embedding_dim == self.config.transformer_hidden_dim
        if self.mode == 'spatial': # treat the timestep axis as a minibatch axis
            x = x.view(batch_size*timesteps, tokens_per_timestep, embedding_dim)
        elif self.mode == 'temporal': # treat the spatial axis as a minibatch axis
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size*tokens_per_timestep, timesteps, embedding_dim)
        else:
            assert False
        eff_batch_size, eff_seq_len, embedding_dim = x.shape
        q, k, v = (
            self.to_qkv(x)
            .view(eff_batch_size, eff_seq_len, 3, self.config.transformer_head_count, self.config.transformer_head_dim)
            .unbind(2)
        )
        if self.mode == 'temporal':
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)
        pre_out = (
            xformers_ops.memory_efficient_attention(q, k, v, attn_bias=xformers_ops.LowerTriangularMask() if self.mode == 'temporal' else None, p=self.config.dropout if self.training else 0.)
            .view(eff_batch_size, eff_seq_len, embedding_dim)
        )
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        if self.mode == 'spatial':
            out = out.view(batch_size, timesteps, tokens_per_timestep, embedding_dim)
        elif self.mode == 'temporal':
            out = out.view(batch_size, tokens_per_timestep, timesteps, embedding_dim).permute(0, 2, 1, 3).contiguous()
        else:
            assert False
        return out

# Maps the spatial tokens @ each time to a prediction.
#  Note that the tokens at time t depend on all previous tokens via the preceding temporal attention layers.
class AttentionPool(BaseModule):
    def __init__(self, config: Config, query_count: int = 1):
        self.config = config
        self.query_count = query_count
        super().__init__()
    
    def construct(self):
        self.q = nn.Parameter(torch.empty((self.query_count, self.config.transformer_hidden_dim,), dtype=torch.float))
        self.to_kv = nn.Linear(self.config.transformer_hidden_dim, 2*self.config.transformer_hidden_dim)
        self.to_out = nn.Linear(self.config.transformer_hidden_dim, self.config.transformer_hidden_dim)
        self.out_dropout = nn.Dropout(self.config.dropout)
    
    def init_weights(self):
        nn.init.normal_(self.q, mean=0., std=0.02)
        nn.init.trunc_normal_(self.to_kv.weight, mean=0., std=0.02)
        nn.init.constant_(self.to_kv.bias, 0)
        nn.init.trunc_normal_(self.to_out.weight, mean=0., std=0.02)
        nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, timesteps, tokens_per_timestep, embedding_dim = x.shape
        x = x.view(batch_size*timesteps, tokens_per_timestep, embedding_dim)
        eff_batch_size, eff_seq_len, embedding_dim = x.shape
        q = (
            self.q
            .view(1, self.query_count, embedding_dim)
            .expand(eff_batch_size, -1, -1)
            .view(eff_batch_size, self.query_count, self.config.transformer_head_count, self.config.transformer_head_dim)
        )
        k, v = (
            self.to_kv(x)
            .view(eff_batch_size, eff_seq_len, 2, self.config.transformer_head_count, self.config.transformer_head_dim)
            .unbind(2)
        )
        q = q.to(k.dtype)
        pre_out = (
            xformers_ops.memory_efficient_attention(q, k, v, p=self.config.dropout if self.training else 0)
            .view(eff_batch_size, self.query_count, embedding_dim)
        )
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        out = out.view(batch_size, timesteps, self.query_count, embedding_dim)
        return out