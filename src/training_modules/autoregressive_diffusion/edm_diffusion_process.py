from typing import Sequence, Literal
from math import sqrt

import numpy as np
import torch
from torch import nn

from .config import DiffusionConfig

class EDMSampler(nn.Module):
    dummy_variable: torch.Tensor

    def __init__(self, config: DiffusionConfig, score_estimator: nn.Module):
        super().__init__()
        self.config = config
        self.score_estimator = score_estimator
        self.register_buffer('dummy_variable', torch.tensor(0.))
    
    def get_device(self):
        return self.dummy_variable.device
    
    def get_dtype(self):
        return self.dummy_variable.dtype
    
    def sample_training_noise_level(self, batch_size: int) -> torch.Tensor:
        log_sigma = self.dummy_variable.new_tensor(torch.randn(batch_size))
        log_sigma = self.config.p_std*log_sigma + self.config.p_mean
        return log_sigma.exp()
    
    def get_training_loss_weight(self, noise_level: torch.Tensor):
        return (noise_level**2 + self.config.sigma_data**2)/((noise_level*self.config.sigma_data)**2)
    
    def get_sigma_sequence(self, count: int, direction: Literal['increasing', 'decreasing']) -> torch.Tensor:
        noise_levels = [
            (
                self.config.sigma_max**(1/self.config.rho)
                + (i/(count-2))*(self.config.sigma_min**(1/self.config.rho) - self.config.sigma_max**(1/self.config.rho))
            )**self.config.rho for i in range(count-1)
        ] + [0.0]
        if direction == 'increasing':
            noise_levels = noise_levels[::-1]
        elif direction == 'decreasing':
            pass
        else:
            assert False
        noise_levels = self.dummy_variable.new_tensor(torch.tensor(noise_levels))
        return noise_levels
    
    def get_gamma_sequence(self, sigma_sequence: torch.Tensor) -> torch.Tensor:
        count = len(sigma_sequence)
        gamma_sequence = [
            min(self.config.s_churn/count, sqrt(2)-1) if self.config.s_tmin <= t <= self.config.s_tmax else 0. for t in sigma_sequence
        ]
        gamma_sequence = self.dummy_variable.new_tensor(torch.tensor(gamma_sequence))
        return gamma_sequence
    
    def get_input_scaling_factor(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1/(sigma**2 + self.config.sigma_data**2).sqrt()
    
    def get_noise_conditioning(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25*sigma.log()
    
    def sample_from_ode(self, shape: Sequence[int], timestep_count: int) -> torch.Tensor:
        sigma_sequence = self.get_sigma_sequence(timestep_count, 'decreasing')
        x_i = self.dummy_variable.new_tensor(sigma_sequence[0]*torch.randn(*shape))
        for i in range(timestep_count-1):
            sigma_i = sigma_sequence[i]
            sigma_i1p = sigma_sequence[i+1]
            d_i = -sigma_i*self.score_estimator(self.get_input_scaling_factor(sigma_i)*x_i, self.get_noise_conditioning(sigma_i))
            x_i1p = x_i + (sigma_i1p - sigma_i)*d_i
            if sigma_i1p > 0:
                d_ip = -sigma_i1p*self.score_estimator(self.get_input_scaling_factor(sigma_i1p)*x_i1p, self.get_noise_conditioning(sigma_i1p))
                x_i1p = x_i + 0.5*(sigma_i1p - sigma_i)*(d_i + d_ip)
            x_i = x_i1p
        return x_i
    
    def sample_from_sde(self, shape: Sequence[int], timestep_count: int) -> torch.Tensor:
        sigma_sequence = self.get_sigma_sequence(timestep_count, 'decreasing')
        gamma_sequence = self.get_gamma_sequence(sigma_sequence)
        x_i = self.dummy_variable.new_tensor(sigma_sequence[0]*torch.randn(*shape))
        for i in range(timestep_count-1):
            sigma_i = sigma_sequence[i]
            sigma_i1p = sigma_sequence[i+1]
            gamma_i = gamma_sequence[i]
            t_ih = (1 + gamma_i)*sigma_i
            x_ih = x_i + self.config.s_noise*(t_ih**2 - sigma_i**2).sqrt()*torch.randn_like(x_i)
            d_i = -sigma_i*self.score_estimator(self.get_input_scaling_factor(t_ih)*x_ih, self.get_noise_conditioning(t_ih))
            x_i1p = x_ih + (sigma_i1p - t_ih)*d_i
            if sigma_i1p > 0:
                d_ip = -sigma_i1p*self.score_estimator(self.get_input_scaling_factor(sigma_i1p)*x_i1p, self.get_noise_conditioning(sigma_i1p))
                x_i1p = x_ih + 0.5*(sigma_i1p - t_ih)*(d_i + d_ip)
            x_i = x_i1p
        return x_i