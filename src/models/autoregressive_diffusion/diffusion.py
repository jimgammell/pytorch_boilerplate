from typing import Union, Literal, Optional
from math import cos, pi, sqrt

from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
import torchsde
import torch
from torch import nn

from .score_estimator import ScoreEstimator

class DiffusionSimulator(nn.Module):
    def __init__(self,
        score_estimator: ScoreEstimator,
        max_time: float = 1.0,
        max_temp: float = 1e1,
        min_temp: float = 0.5,
        temp_scheduler: Literal['linear', 'exponential', 'cosine'] = 'linear'
    ):
        super().__init__()
        self.score_estimator = score_estimator
        self.max_time = max_time
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.temp_scheduler = temp_scheduler

    def get_temperature(self, t: float) -> float:
        if self.temp_scheduler == 'linear':
            return self.max_temp + (self.min_temp - self.max_temp)*(t/self.max_time)
        elif self.temp_scheduler == 'exponential':
            return self.max_temp*(self.min_temp/self.max_temp)**(t/self.max_time)
        elif self.temp_scheduler == 'cosine':
            return (self.max_temp-self.min_temp)*(0.5*cos(pi*t/self.max_time) + 0.5) + self.min_temp
        else:
            assert False

class OrnsteinUnlenbeckSimulator(DiffusionSimulator):
    theta: torch.Tensor

    def __init__(self,
        score_estimator: ScoreEstimator,
        theta: float,
        vocab_size: int,
        max_time: float = 1.0,
        max_temp: float = 1e1,
        min_temp: float = 1e-1,
        temp_scheduler: Literal['linear', 'exponential', 'cosine'] = 'linear'
    ):
        super().__init__(score_estimator, max_time=max_time, max_temp=max_temp, min_temp=min_temp, temp_scheduler=temp_scheduler)
        self.vocab_size = vocab_size
        self.register_buffer('theta', torch.tensor(theta, dtype=torch.float))
    
    def get_device(self):
        return self.theta.device
    
    def sample_latent(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        y = torch.randn(batch_size, sequence_length, self.vocab_size, device=self.get_device())
        x = nn.functional.softmax(y, dim=-1)
        return x
    
    def g_prod(self, t, x, v):
        x = x.reshape(1, -1, self.vocab_size)
        v = v.reshape(1, -1, self.vocab_size)
        rv = x*v - x*(x*v).sum(dim=-1, keepdim=True)
        return rv.reshape(1, -1)
    
    def drift(self, t, x, context):
        x = x.reshape(1, -1, self.vocab_size)
        f = -self.theta*x*(1-x)
        score_estimator_temp = self.get_temperature(t)
        score = self.score_estimator.get_stein_score(x, context=context, temperature=score_estimator_temp)
        score = x*score - x*(x*score).sum(dim=-1, keepdim=True)
        score = x*score - x*(x*score).sum(dim=-1, keepdim=True)
        diffn = x.pow(2).sum(dim=-1, keepdim=True)*x - x*x
        return (f - 0.5*diffn - 0.5*score).reshape(1, -1)
    
    def run_reverse_sde(self, sequence_length: int, step_count: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        class SDE(nn.Module):
            noise_type = 'diagonal'
            sde_type = 'ito'
            def __init__(self, outer_self):
                super().__init__()
                self.outer_self = outer_self
            def f(self, t, y):
                return -self.outer_self.drift(1-t, y, context)
            def g(self, t, y): # this isn't used (hopefully), but has to be passed as part of the argument checking
                return torch.full_like(y, torch.nan)
            def g_prod(self, t, x, v):
                return self.outer_self.g_prod(1-t, x, v)
        sde = SDE(self)
        x_1 = self.sample_latent(1, sequence_length).reshape(1, -1)
        t_int = torch.tensor([0., 1.], dtype=torch.float, device=self.get_device())
        x_0 = torchsde.sdeint(sde, x_1, t_int, method='euler', adaptive=True)
        tokens = x_0.argmax(dim=-1)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        return tokens

class CoxIngersollRossSimulator(DiffusionSimulator):
    alpha: torch.Tensor
    b: torch.Tensor

    def __init__(self,
        score_estimator: ScoreEstimator,
        alpha: float,
        vocab_size: int,
        b: float = 1.0,
        max_time: float = 1.0,
        max_temp: float = 1e1,
        min_temp: float = 0.5,
        temp_scheduler: Literal['linear', 'exponential', 'cosine'] = 'exponential'
    ):
        super().__init__(score_estimator, max_time=max_time, max_temp=max_temp, min_temp=min_temp, temp_scheduler=temp_scheduler)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float))
        self.vocab_size = vocab_size
    
    def get_latent_distribution(self):
        return torch.distributions.Gamma(self.alpha, 1.)
    
    def reverse_sde_euler_rollout(self, batch_size: int, sequence_length: int, step_count: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        step_size = self.max_time/step_count
        latent_distribution = self.get_latent_distribution()
        Y = latent_distribution.sample((batch_size, sequence_length, self.vocab_size))
        for step_idx in tqdm(range(step_count)):
            t = step_idx/(step_count-1)
            score_estimator_temp = self.get_temperature(t)
            dY = ( # Eqn. 13 of Richemond et al.
                (-self.b*(self.alpha - Y) + 2*self.b*Y*self.score_estimator.get_stein_score(Y, context=context, temperature=score_estimator_temp) + 2*self.b)*step_size
                + (2*self.b*Y + 1e-6).sqrt()*torch.randn_like(Y)*sqrt(step_size)
            )
            Y = (Y + dY).clamp_(min=0)
        tokens = Y.argmax(dim=-1)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        return tokens
    
    def _reverse_ode_euler_rollout(self, batch_size: int, sequence_length: int, step_count: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        step_size = self.max_time/step_count
        latent_distribution = self.get_latent_distribution()
        log_Y = latent_distribution.sample((batch_size, sequence_length, self.vocab_size)).log()
        for step_idx in range(step_count):
            t = step_idx/(step_count-1)
            score_estimator_temp = self.get_temperature(t)
            dlog_Y = ( # Eqn. 17 -- log version of Richemond et al.
                -self.b*(1 + self.score_estimator.get_stein_score(log_Y.exp(), context=context, temperature=score_estimator_temp))*step_size
            )
            log_Y = (log_Y + dlog_Y)
        tokens = log_Y.argmax(dim=-1)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        return tokens
    
    def reverse_ode_euler_rollout(self, batch_size: int, sequence_length: int, step_count: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        log_y_i = torch.zeros(batch_size, sequence_length, self.vocab_size, dtype=torch.float, device=self.b.device) #self.get_latent_distribution().sample((batch_size, sequence_length, self.vocab_size)).log()
        device = log_y_i.device
        log_y_i = log_y_i.detach().cpu().numpy()
        shape = log_y_i.shape
        log_y_i = log_y_i.reshape(-1)
        def f(t, log_y):
            score_estimator_temp = self.get_temperature(self.max_time-t)
            log_y = torch.from_numpy(log_y).to(device).to(torch.float).reshape(*shape)
            dlog_Y = -self.b*(1 + self.score_estimator.get_stein_score(log_y.exp(), context=context, temperature=score_estimator_temp))
            dlog_Y = dlog_Y.detach().cpu().numpy().reshape(-1)
            return dlog_Y
        res = solve_ivp(
            f, (self.max_time, 0), log_y_i, method='RK45', t_eval=(0,), rtol=1e-8, atol=1e-8
        )
        log_y_f = torch.from_numpy(res.y).to(device).to(torch.float).reshape(*shape)
        tokens = log_y_f.argmax(dim=-1)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        return tokens