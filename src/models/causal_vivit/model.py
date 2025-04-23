from typing import Optional, Tuple, List

import torch

from common import *
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
        self.to_class_logits = nn.Linear(self.config.transformer_hidden_dim, self.config.out_dim)
        if self.config.sparse_inputs:
            self.to_patch_logits = nn.Linear(self.config.transformer_hidden_dim, self.config.patch_count)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.to_class_logits.weight, mean=0., std=0.02)
        nn.init.constant_(self.to_class_logits.bias, 0)
        if self.config.sparse_inputs:
            nn.init.trunc_normal_(self.to_patch_logits.weight, mean=0., std=0.02)
            nn.init.constant_(self.to_patch_logits.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention_pool(self.pre_attn_pool_norm(x), mask)
        class_logits = self.to_class_logits(x[:, :, 0, :])
        if self.config.sparse_inputs:
            patch_logits = self.to_patch_logits(x[:, :, 1, :])
            return class_logits, patch_logits
        else:
            return class_logits

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
    
    def init_weights(self):
        if self.config.init_from_deit is not None:
            self.init_with_deit_weights(self.config.init_from_deit)
    
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
    
    @staticmethod
    def load_weight(dest: torch.Tensor, src: torch.Tensor):
        assert dest.shape == src.shape, (dest.shape, src.shape)
        dest.data = src

    def init_with_deit_weights(self, model_size: Literal['tiny', 'tiny-distilled', 'small', 'small-distilled', 'base-mae']):
        if model_size == 'tiny':
            weights_filename = r'deit_tiny_patch16_224-a1311bcf.pth'
        elif model_size == 'tiny-distilled':
            weights_filename = r'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
        elif model_size == 'small':
            weights_filename = r'deit_small_patch16_224-cd65a155.pth'
        elif model_size == 'small-distilled':
            weights_filename = r'deit_small_distilled_patch16_224-649709d9.pth'
        elif model_size == 'base-mae':
            weights_filename = r'mae_pretrain_vit_base.pth'
        else:
            assert False
        weights_path = os.path.join(RESOURCE_DIR, 'deit_pretrained_models', weights_filename)
        weights = torch.load(weights_path)['model']
        self.load_weight(self.patchifier.patch_embedders[0].weight, weights['patch_embed.proj.weight'])
        self.load_weight(self.patchifier.patch_embedders[0].bias, weights['patch_embed.proj.bias'])
        self.load_weight(
            self.patchifier.spatial_position_embedding[:, :, :self.config.patch_count_without_downsampling, :],
            nn.functional.interpolate(
                weights['pos_embed'].permute(0, 2, 1), size=self.config.patch_count_without_downsampling, mode='linear'
            ).permute(0, 2, 1).contiguous().view(1, 1, self.config.patch_count_without_downsampling, self.config.transformer_hidden_dim)
        )
        for layer_idx, layer in enumerate(self.transformer_layers):
            nn.init.constant_(layer.temporal_attention.to_out.weight, 0)
            nn.init.constant_(layer.temporal_attention.to_out.bias, 0)
            self.load_weight(layer.spatial_attention.to_qkv.weight, weights[f'blocks.{layer_idx}.attn.qkv.weight'])
            self.load_weight(layer.spatial_attention.to_qkv.bias, weights[f'blocks.{layer_idx}.attn.qkv.bias'])
            self.load_weight(layer.spatial_attention.to_out.weight, weights[f'blocks.{layer_idx}.attn.proj.weight'])
            self.load_weight(layer.spatial_attention.to_out.bias, weights[f'blocks.{layer_idx}.attn.proj.bias'])
            self.load_weight(layer.pre_spatial_attn_norm.norm_layer.weight, weights[f'blocks.{layer_idx}.norm1.weight'])
            self.load_weight(layer.pre_spatial_attn_norm.norm_layer.bias, weights[f'blocks.{layer_idx}.norm1.bias'])
            self.load_weight(layer.fnn.fc1.weight, weights[f'blocks.{layer_idx}.mlp.fc1.weight'])
            self.load_weight(layer.fnn.fc1.bias, weights[f'blocks.{layer_idx}.mlp.fc1.bias'])
            self.load_weight(layer.fnn.fc2.weight, weights[f'blocks.{layer_idx}.mlp.fc2.weight'])
            self.load_weight(layer.fnn.fc2.bias, weights[f'blocks.{layer_idx}.mlp.fc2.bias'])

class SparseInputTransformer(Transformer):
    def construct(self):
        super().construct()
        self.patch_selector = PatchSelector(self.config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patchifier(x)
        batch_size, timestep_count, patch_count, embedding_dim = x.shape
        input = torch.full((batch_size, timestep_count, self.config.per_frame_patch_count, embedding_dim), torch.nan, dtype=x.dtype, device=x.device)
        class_logits = torch.full((batch_size, timestep_count, self.config.out_dim), torch.nan, dtype=x.dtype, device=x.device)
        patch_logits = torch.full((batch_size, timestep_count, self.config.patch_count), torch.nan, dtype=x.dtype, device=x.device)
        for time_idx in range(timestep_count):
            input[:, time_idx, :, :] = self.patch_selector(x[:, 0, :, :], patch_logits=patch_logits[:, time_idx-1, :] if time_idx > 0 else None)
            hidden_acts = input
            for layer in self.transformer_layers:
                hidden_acts = layer(hidden_acts, mask)
            new_class_logits, new_patch_logits = self.head(hidden_acts, mask)
            class_logits[:, time_idx, :] = new_class_logits[:, time_idx, :]
            patch_logits[:, time_idx, :] = new_patch_logits[:, time_idx, :]
        return class_logits