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
        #assert (sequence_logits >= 0).all() # Should look into this -- might force GPU-CPU sync
        sequence_probs = sequence_logits
        embedded_sequence = torch.matmul(sequence_probs, self.embedding_dict)
        if context is None:
            context = torch.tensor(self.arm.token_encoder.eot_token, dtype=torch.long, device=sequence_logits.device).view(1, 1).expand(batch_size, 1)
        embedded_context = self.arm.token_embedding(context)
        embedded_sequence = torch.cat([embedded_context, embedded_sequence], dim=1)
        position_tokens = torch.arange(0, embedded_sequence.size(1), device=embedded_sequence.device, dtype=torch.long)
        embedded_sequence = embedded_sequence + self.arm.position_embedding(position_tokens)
        embedded_sequence = self.arm.trunk(embedded_sequence)
        logits = self.arm.head(embedded_sequence)
        logits = logits[:, context.size(1)-1:-1, :]
        log_likelihood = (
            nn.functional.softmax(sequence_logits, dim=-1) * nn.functional.log_softmax(logits/temperature, dim=-1)
        ).sum(dim=-1).mean()
        print(f'log_likelihood: {log_likelihood.item()}, entropy: {-(nn.functional.softmax(sequence_logits, dim=-1)*nn.functional.log_softmax(sequence_logits, dim=-1)).sum(dim=-1).mean().item()}, max_prob: {nn.functional.softmax(sequence_logits, dim=-1).max(dim=-1).values.mean().item()}')
        return log_likelihood
    
    def get_stein_score(self, sequence_logits: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        sequence_logits = sequence_logits.clone()
        sequence_logits.requires_grad_(True)
        sequence_logits.grad = None
        sequence_log_likelihood = self.get_log_likelihood(sequence_logits, context=context, temperature=temperature)
        rv = torch.autograd.grad(sequence_log_likelihood, sequence_logits)
        assert rv is not None
        score = rv[0].detach()
        return score