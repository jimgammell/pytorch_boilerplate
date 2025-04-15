from dataclasses import dataclass

@dataclass
class Config:
    input_channel_count: int = 3
    input_dim: int = 256 # transformer inputs will have shape (batch_size, self.input_channel_count, self.input_dim, self.input_dim)
    output_dim: int = 10 # number of dimensions in the output head -- e.g. 10 if we are doing 10-class classification
    patch_size: int = 16
    embedding_dim: int = 768 # hidden activation dimension
    mlp_dim: int = 3072
    transformer_layer_count: int = 3 # number of transformer layers
    attn_head_count: int = 12 # number of heads for multiheaded attention
    bias: bool = False # whether to use biases in layers where it is appropriate
    dropout: float = 0.1 # whether/how much to use dropout where appropriate
    train_prior_prob: float = 0.05 # percent of the time we should train the model's prior patch probability instead of next patch predictor
    gumbel_tau: float = 1.0 # temperature of the Gumbel softmax distribution
    eps: float = 1e-5 # constant to avoid dividing by zero

    def __post_init__(self):
        assert isinstance(self.input_channel_count, int) and (self.input_channel_count > 0)
        assert isinstance(self.input_dim, int) and (self.input_dim > 0)
        assert isinstance(self.output_dim, int) and (self.output_dim > 0)
        assert isinstance(self.embedding_dim, int) and (self.embedding_dim > 0)
        assert isinstance(self.mlp_dim, int) and (self.mlp_dim > 0)
        assert isinstance(self.attn_head_count, int) and (1 <= self.attn_head_count <= self.embedding_dim) and (self.embedding_dim % self.attn_head_count == 0)
        assert isinstance(self.bias, bool)
        assert isinstance(self.dropout, float) and (0 <= self.dropout < 1)
        assert isinstance(self.train_prior_prob, float) and (0 <= self.train_prior_prob < 1)
        assert isinstance(self.gumbel_tau, float) and (0 < self.gumbel_tau < float('inf'))
        assert isinstance(self.eps, float) and (0 < self.eps < float('inf'))
        assert self.input_dim % self.patch_size == 0
        self.patch_count = (self.input_dim//self.patch_size)**2
        self.attn_head_dim = self.embedding_dim // self.attn_head_count