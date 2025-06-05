from typing import Optional

import torch
from torch import nn

from .arm import ARM

class ScoreEstimator(nn.Module):
    def __init__(self, arm: ARM):
        super().__init__()

        assert isinstance(arm, ARM)
        for param in arm.parameters():
            param.requires_grad_(False)
        self.arm = arm
        self.embedding_dict = self.arm.token_embedding.weight
        self.token_dict_size, self.embedding_dim = self.embedding_dict.shape
    
    def get_log_likelihood(self, sequence_logits: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        if temperature is None:
            temperature = 1.0
        else:
            assert 0 < temperature < float('inf')
        batch_size, sequence_length, class_count = sequence_logits.shape
        assert class_count == self.token_dict_size
        embedded_sequence = nn.functional.softmax(sequence_logits, dim=-1) @ self.embedding_dict
        if context is not None:
            embedded_context = self.arm.token_embedding(context)
            embedded_sequence = torch.cat([embedded_context, embedded_sequence], dim=1)
        embedded_sequence = self.arm.trunk(embedded_sequence)
        logits = self.arm.head(embedded_sequence)[:, -sequence_length-1:-1, :]
        log_likelihood = (
            nn.functional.softmax(sequence_logits, dim=-1) * nn.functional.log_softmax(logits/temperature, dim=-1)
        ).sum(dim=-1).mean()
        return log_likelihood
    
    def get_stein_score(self, sequence_logits: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        sequence_logits = sequence_logits.clone()
        sequence_logits.requires_grad_(True)
        sequence_log_likelihood = self.get_log_likelihood(sequence_logits, context=context, temperature=temperature)
        rv = torch.autograd.backward(sequence_log_likelihood, inputs=sequence_logits)
        assert rv is not None
        score = rv[0]
        return score.detach()