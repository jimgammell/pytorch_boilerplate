from typing import Optional
from math import log2
from dataclasses import dataclass

import numpy as np

@dataclass
class Config:
    input_channel_count: int = 3
    input_dim: int = 224 # transformer inputs will have shape (batch_size, self.input_channel_count, self.input_dim, self.input_dim)
    output_dim: int = 10 # number of dimensions in the output head -- e.g. 10 if we are doing 10-class classification
    base_patch_dim: int = 14 # width == height of the smallest patches.
    resolutions: int = 4 # Number of times to apply 2x average pooling to the image and re-patchify.
    embedding_dim: int = 768 # hidden activation dimension
    transformer_layer_count: int = 3 # number of transformer layers
    attn_head_count: int = 12 # number of heads for multiheaded attention
    bias: bool = False # whether to use biases in layers where it is appropriate
    dropout: float = 0.1 # whether/how much to use dropout where appropriate
    train_prior_prob: float = 0.05 # percent of the time we should train the model's prior patch probability instead of next patch predictor
    gumbel_tau: float = 1.0 # temperature of the Gumbel softmax distribution
    eps: float = 1e-5 # constant to avoid dividing by zero
    max_sequence_length: Optional[int] = None # maximum number of patches the model can look at

    def __post_init__(self):
        assert isinstance(self.input_channel_count, int) and (self.input_channel_count > 0)
        assert isinstance(self.input_dim, int) and (self.input_dim > 0)
        assert isinstance(self.output_dim, int) and (self.output_dim > 0)
        assert isinstance(self.base_patch_dim, int) and (self.base_patch_dim > 0) and (self.input_dim % self.base_patch_dim == 0)
        assert isinstance(self.resolutions, int) and (1 <= self.resolutions <= log2(self.input_dim//self.base_patch_dim))
        assert isinstance(self.embedding_dim, int) and (self.embedding_dim > 0)
        assert isinstance(self.attn_head_count, int) and (1 <= self.attn_head_count <= self.embedding_dim) and (self.embedding_dim % self.attn_head_count == 0)
        assert isinstance(self.bias, bool)
        assert isinstance(self.dropout, float) and (0 <= self.dropout < 1)
        assert isinstance(self.train_prior_prob, float) and (0 <= self.train_prior_prob < 1)
        assert isinstance(self.gumbel_tau, float) and (0 < self.gumbel_tau < float('inf'))
        assert isinstance(self.eps, float) and (0 < self.eps < float('inf'))
        self.patch_dim = self.input_channel_count*self.base_patch_dim**2
        self.per_res_patch_counts = np.array([(self.input_dim//(self.base_patch_dim*2**downsample_count))**2 for downsample_count in range(self.resolutions)], dtype=int)
        self.patch_count = self.per_res_patch_counts.sum()
        if self.max_sequence_length is None:
            self.max_sequence_length = self.patch_count
        self.attn_head_dim = self.embedding_dim // self.attn_head_count
        assert isinstance(self.max_sequence_length, int) and (0 < self.max_sequence_length <= self.patch_count)