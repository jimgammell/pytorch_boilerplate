from random import choices
import torch
from torch import nn

from .soft_xor import soft_xor

class ASCADv1_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout1d(p=0.0)

    def forward(self, x):
        batch_size, token_count, embedding_dim = x.shape
        assert token_count == 65 # predicting SubBytes, SubBytes_r, SubBytes_rout, r, rout
        subbytes_logits, subbytes_r_logits, subbytes_rout_logits, r_logits, rout_logits = x.split((16, 16, 16, 16, 1), dim=1)
        subbytes_logits_from_r = soft_xor(subbytes_r_logits, r_logits)
        subbytes_logits_from_rout = soft_xor(subbytes_rout_logits, rout_logits)
        logits = torch.stack([subbytes_logits, subbytes_logits_from_r, subbytes_logits_from_rout], dim=2)
        logits = self.dropout(logits.reshape(batch_size*16, 3, embedding_dim)).reshape(batch_size, 16, 3, embedding_dim)
        logits = logits.mean(dim=2)
        return logits