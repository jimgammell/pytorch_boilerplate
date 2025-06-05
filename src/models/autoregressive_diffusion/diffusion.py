from typing import Union, Literal
from math import cos, pi

import numpy as np
import torch
from torch import nn

from .score_estimator import ScoreEstimator

class CoxIngersollRossSimulator(nn.Module):
    alpha: torch.Tensor
    b: torch.Tensor

    def __init__(self,
        score_estimator: ScoreEstimator,
        alpha: Union[float, np.ndarray],
        vocab_size: int,
        max_time: float = 1.0,
        max_temp: float = 1e2,
        min_temp: float = 1e-2,
        temp_scheduler: Literal['linear', 'exponential', 'cosine'] = 'exponential'
    ):
        super().__init__()
        self.score_estimator = score_estimator
        if isinstance(alpha, float):
            alpha = np.full((vocab_size,), alpha)
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape == (vocab_size,)
        else:
            assert False
        self.register_buffer('alpha', torch.from_numpy(alpha))
        self.register_buffer('b', torch.tensor(1.))
        self.latent_distribution = torch.distributions.Dirichlet(self.alpha)
        self.vocab_size = vocab_size
        self.max_time = max_time
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.temp_scheduler = temp_scheduler
    
    def get_temperature(self, step_idx: int, step_count: int) -> float:
        if self.temp_scheduler == 'linear':
            return self.max_temp + (self.min_temp - self.max_temp)*(step_idx/(step_count-1))
        elif self.temp_scheduler == 'exponential':
            return self.max_temp*(self.min_temp/self.max_temp)**(step_idx/(step_count-1))
        elif self.temp_scheduler == 'cosine':
            return (self.max_temp-self.min_temp)*(0.5*cos(pi*step_idx/(step_count-1)) + 0.5) + self.min_temp
        else:
            assert False
    
    def reverse_sde_euler_rollout(self, batch_size: int, sequence_length: int, step_count: int) -> torch.Tensor:
        step_size = self.max_time/step_count
        token_dist = self.latent_distribution.sample((batch_size, sequence_length))
        for step_idx in range(step_count):
            score_estimator_temp = self.get_temperature(step_idx, step_count)
            token_dist = ( # Eqn. 13 of Richemond et al.
                (-self.b*(self.alpha - token_dist) + 2*self.b*token_dist*self.score_estimator.get_score(token_dist, temperature=score_estimator_temp) + 2*self.b)*step_size
                + (2*self.b*token_dist).sqrt()*torch.randn_like(token_dist)
            )
        return token_dist
    
    def reverse_ode_euler_rollout(self, batch_size: int, sequence_length: int, step_count: int) -> torch.Tensor:
        step_size = self.max_time/step_count
        token_dist = self.latent_distribution.sample((batch_size, sequence_length))
        for step_idx in range(step_count):
            score_estimator_temp = self.get_temperature(step_idx, step_count)
            token_dist = ( # Eqn. 17 of Richemond et al.
                self.b*(self.alpha - 1 - token_dist - token_dist*self.score_estimator.get_score(token_dist, temperature=score_estimator_temp))*step_size
            )
        return token_dist