from typing import Literal, Optional

import torch
from torch import nn
from torchvision.ops import MLP

from ..base_module import BaseModule
from .config import Config

class Patchifier(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.patch_embedder = nn.Conv2d(
            self.config.input_channels, self.config.transformer_hidden_dim, kernel_size=self.config.patch_dim, stride=self.config.patch_dim
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.config.max_input_temporal_dim, self.config.patch_count, self.config.transformer_hidden_dim))
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.patch_embedder.weight)
        nn.init.constant_(self.patch_embedder.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape
        assert timesteps == self.config.max_input_temporal_dim
        assert channels == self.config.input_channels
        assert height == width == self.config.input_spatial_dim
        x = x.view(batch_size*timesteps, channels, height, width)
        x = self.patch_embedder(x)
        x = x.view(batch_size, timesteps, self.config.transformer_hidden_dim, -1)
        x = x.permute(0, 1, 3, 2)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x

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
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
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
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.constant_(self.to_qkv.bias, 0)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, timesteps, tokens_per_timestep, embedding_dim = x.shape
        assert timesteps == self.config.max_input_temporal_dim
        assert embedding_dim == self.config.transformer_hidden_dim
        if self.mode == 'spatial':
            q, k, v = (
                self.to_qkv(x)
                .reshape(batch_size, timesteps, tokens_per_timestep, 3, self.config.transformer_head_count, self.config.transformer_head_dim)
                .permute(3, 0, 1, 4, 2, 5)
                .unbind(0)
            )
        elif self.mode == 'temporal':
            assert mask is None
            q, k, v = (
                self.to_qkv(x)
                .reshape(batch_size, timesteps, tokens_per_timestep, 3, self.config.transformer_head_count, self.config.transformer_head_dim)
                .permute(3, 0, 2, 4, 1, 5)
                .unbind(0)
            )
        else:
            assert False
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.config.dropout, is_causal=self.mode=='temporal')
        if self.mode == 'spatial':
            pre_out = (
                pre_out
                .permute(0, 1, 3, 2, 4)
                .reshape(batch_size, timesteps, tokens_per_timestep, self.config.transformer_hidden_dim)
            )
        elif self.mode == 'temporal':
            pre_out = (
                pre_out
                .permute(0, 3, 1, 2, 4)
                .reshape(batch_size, timesteps, tokens_per_timestep, self.config.transformer_hidden_dim)
            )
        else:
            assert False
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        return out

# Maps the spatial tokens @ each time to a prediction.
#  Note that the tokens at time t depend on all previous tokens via the preceding temporal attention layers.
class AttentionPool(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.q = nn.Parameter(torch.zeros(self.config.transformer_hidden_dim))
        self.to_kv = nn.Linear(self.config.transformer_hidden_dim, 2*self.config.transformer_hidden_dim)
        self.to_out = nn.Linear(self.config.transformer_hidden_dim, self.config.transformer_hidden_dim)
        self.out_dropout = nn.Dropout(self.config.dropout)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.q, std=1)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.constant_(self.to_kv.bias, 0)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, timesteps, tokens_per_timestep, embedding_dim = x.shape
        q = (
            self.q
            .view(1, 1, 1, embedding_dim)
            .expand(batch_size, timesteps, -1, -1)
            .view(batch_size, timesteps, 1, self.config.transformer_head_count, self.config.transformer_head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        k, v = (
            self.to_kv(x)
            .reshape(batch_size, timesteps, tokens_per_timestep, 2, self.config.transformer_head_count, self.config.transformer_head_dim)
            .permute(3, 0, 1, 4, 2, 5)
            .unbind(0)
        )
        pre_out = (
            nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.config.dropout if self.training else 0)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_size, timesteps, self.config.transformer_hidden_dim)
        )
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        return out