from random import choices
import torch
from torch import nn

from .soft_xor import soft_xor

class ASCADv1_Head(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pre_out = nn.Linear(self.embedding_dim, 256, bias=False) # disabling bias === uniform prior over byte values
        nn.init.xavier_uniform_(self.pre_out.weight)
    
    def forward(self, x):
        batch_size, token_count, embedding_dim = x.shape
        assert embedding_dim == self.embedding_dim
        assert token_count == 65 # predicting SubBytes, SubBytes_r, SubBytes_rout, r, rout
        x = self.pre_out(x)
        subbytes_logits, subbytes_r_logits, subbytes_rout_logits, r_logits, rout_logits = x.split((16, 16, 16, 16, 1), dim=1)
        subbytes_logits_from_r = soft_xor(subbytes_r_logits, r_logits)
        subbytes_logits_from_rout = soft_xor(subbytes_rout_logits, rout_logits)
        logits = torch.stack([subbytes_logits, subbytes_logits_from_r, subbytes_logits_from_rout], dim=0)
        if self.training:
            indices = torch.tensor(
                choices([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]], k=batch_size),
            dtype=torch.float32).reshape(batch_size, 3, 1, 1)
            logits = (indices.sum(dim=1, keepdim=True)/3)*logits*indices
            logits = logits.mean(dim=0)
        else:
            logits = logits.mean(dim=0)
        return logits