from typing import Optional, Tuple, List

import torch
from torch import nn

from common import *
from ..base_module import BaseModule
from .config import Config
from .building_blocks import *

class TransformerLayer(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
    
    def construct(self):
        self.norm1 = Norm(self.config)
        self.attn = Attention(self.config)
        self.norm2 = Norm(self.config)
        self.fnn = FeedForward(self.config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fnn(self.norm2(x))
        return x

class Head(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
    
    def construct(self):
        self.attn_pool = AttentionPool(self.config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.to_logits = nn.Linear(self.config.embedding_dim, self.config.output_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        x = self.attn_pool(x)
        x = self.dropout(x)
        x = self.to_logits(x).reshape(batch_size, self.config.output_dim)
        return x

class Transformer(BaseModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.construct()
        self.init_weights()
    
    def construct(self):
        self.position_embedding = nn.Parameter(0.02*torch.randn(1, self.config.patch_count, self.config.embedding_dim))
        self.patch_embedder = PatchEmbedder(self.config)
        self.transformer_layers = nn.ModuleList(
            TransformerLayer(self.config) for _ in range(self.config.transformer_layer_count)
        )
        self.head = Head(self.config)

    def init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x) + self.position_embedding
        batch_size, token_count, embedding_dim = x.shape
        assert token_count == self.config.patch_count
        assert embedding_dim == self.config.embedding_dim
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.head(x)
        return x

    def get_params_based_on_should_weight_decay(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        all_params = set(self.parameters())
        yes_decay = set()
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                yes_decay.add(mod.weight)
        no_decay = all_params - yes_decay
        return list(yes_decay), list(no_decay)