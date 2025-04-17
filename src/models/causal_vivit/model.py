from typing import Optional, Tuple, List

import torch

from ..base_module import BaseModule
from .config import Config
from .building_blocks import *

class TransformerBlock(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.pre_spatial_attn_norm = NormLayer(self.config)
        self.spatial_attention = Attention(self.config, mode='spatial')
        self.pre_temporal_attn_norm = NormLayer(self.config)
        self.temporal_attention = Attention(self.config, mode='temporal')
        self.pre_fnn_norm = NormLayer(self.config)
        self.fnn = FeedForward(self.config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.spatial_attention(self.pre_spatial_attn_norm(x), mask)
        x = x + self.temporal_attention(self.pre_temporal_attn_norm(x))
        x = x + self.fnn(self.pre_fnn_norm(x))
        return x

class Head(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.pre_attn_pool_norm = NormLayer(self.config)
        self.attention_pool = AttentionPool(self.config)
        self.to_logits = nn.Linear(self.config.transformer_hidden_dim, self.config.out_dim)
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.constant_(self.to_logits.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention_pool(self.pre_attn_pool_norm(x), mask)
        logits = self.to_logits(x)
        return logits

class Transformer(BaseModule):
    def __init__(self, config: Config):
        self.config = config
        super().__init__()
    
    def construct(self):
        self.patchifier = Patchifier(self.config)
        self.transformer_layers = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.transformer_layer_count)
        )
        self.head = Head(self.config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patchifier(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)
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