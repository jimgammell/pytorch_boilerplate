from collections import OrderedDict
from typing import Optional
from math import sqrt

import tiktoken
from torch import nn

from .arm_building_blocks import *

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn_norm = Norm(self.config)
        self.attn = Attention(self.config)
        self.fnn_norm = Norm(self.config)
        self.fnn = FeedForward(self.config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.fnn(self.fnn_norm(x))
        return x

class Trunk(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer_blocks = nn.ModuleList(TransformerBlock(self.config) for _ in range(self.config.layer_count))
    
    def forward(self, x):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x

class ARM(nn.Module):
    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    trunk: Trunk
    head: nn.Module

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        if self.config.pretrained_model_path is not None:
            self.config.vocab_size = 50257
            self.config.block_size = 1024
            self.config.bias = True
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.trunk = Trunk(self.config)
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.embedding_dim)
        self.head = nn.Sequential(OrderedDict([('norm', Norm(self.config)), ('dense', nn.Linear(self.config.embedding_dim, self.config.vocab_size, bias=False))]))
        self.token_embedding.weight = self.head.dense.weight
        self.apply(self._init_weights)
        for param_name, param in self.named_parameters():
            if param_name.endswith('attn.to_out.weight') or param_name.endswith('fnn.dense_2.weight'):
                nn.init.normal_(param, mean=0., std=0.02/sqrt(2*self.config.layer_count))
        if self.config.pretrained_model_path is not None:
            self._load_pretrained_model(self.config.pretrained_model_path)
        self.token_encoder = tiktoken.get_encoding('gpt2')
    
    def string_to_tokens(self, x: str) -> torch.Tensor:
        return torch.tensor([
            self.token_encoder.encode(x, allowed_special={'<|endoftext|>'})
        ], dtype=torch.long).view(-1).unsqueeze(0)
    
    def tokens_to_string(self, x: torch.Tensor) -> str:
        return self.token_encoder.decode(x[0].tolist())
    
    def _init_weights(self, mod: nn.Module):
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, mean=0., std=0.02)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0., std=0.02)
    
    def _load_pretrained_model(self, path: str):
        hf_state_dict = torch.load(path)
        state_dict = {}
        for k, v in hf_state_dict.items():
            if k.endswith('.attn.masked_bias'):
                continue
            if k.endswith('.attn.bias'):
                continue
            if any(k.endswith(s) for s in {'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'}):
                v = v.t()
            k = k.replace('transformer.wte', 'token_embedding')
            k = k.replace('transformer.wpe', 'position_embedding')
            k = k.replace('transformer.h', 'trunk.transformer_blocks')
            k = k.replace('ln_1', 'attn_norm')
            k = k.replace('ln_2', 'fnn_norm')
            k = k.replace('mlp', 'fnn')
            k = k.replace('attn.c_attn', 'attn.to_qkv')
            k = k.replace('attn.c_proj', 'attn.to_out')
            k = k.replace('fnn.c_fc', 'fnn.dense_1')
            k = k.replace('fnn.c_proj', 'fnn.dense_2')
            k = k.replace('transformer.ln_f', 'head.norm')
            k = k.replace('lm_head', 'head.dense')
            k = k.replace('transformer.', '')
            state_dict[k] = v
        self.load_state_dict(state_dict, strict=True)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.dtype == torch.long
        batch_size, sequence_length = tokens.shape
        device = tokens.device
        position_tokens = torch.arange(0, sequence_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        embedded_tokens = self.dropout(self.token_embedding(tokens) + self.position_embedding(position_tokens))
        latents = self.trunk(embedded_tokens)
        logits = self.head(latents)
        return logits
    
    def autoregressive_sample(self, context: torch.Tensor, max_new_tokens: int, temperature=1.0, top_k=50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = context if context.size(1) < self.config.block_size else context[:, -self.config.block_size:]
            logits = self(context)
            logits = logits[:, -1, :]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            next_token_cpmf = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(next_token_cpmf, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
        return context