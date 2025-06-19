import torch
from torch import nn

from ..generic_trial import Trial
from .config import Config
from models.autoregressive_diffusion.arm import ARM
from models.autoregressive_diffusion.score_estimator import ScoreEstimator
from models.autoregressive_diffusion.diffusion import CoxIngersollRossSimulator, OrnsteinUnlenbeckSimulator

class TextGenerationTrial(Trial):
    def __init__(self, output_dir: str, config: Config):
        super().__init__(output_dir)

        self.config = config
        self.autoregressive_sample_dir = self.register_subtrial_dir('autoregressive_samples')
        self.sde_sample_dir = self.register_subtrial_dir('sde_samples')
        self.ode_sample_dir = self.register_subtrial_dir('ode_samples')
        self.loss_landscape_dir = self.register_subtrial_dir('loss_landscape')
        self.register_experiment_method('generate_ode_samples', self.generate_ode_samples)
        self.register_experiment_method('generate_autoregressive_samples', self.generate_autoregressive_samples)
        self.register_experiment_method('loss_landscape_exploration', self.explore_loss_landscape)

    def construct_arm(self):
        arm = ARM(self.config.arm_config)
        return arm

    @torch.no_grad()
    def generate_autoregressive_samples(self):
        subdir = self.autoregressive_sample_dir()
        arm = self.construct_arm()
        arm.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        arm = arm.to(device)
        base_str = 'My name is Donald Trump.'
        base_tokens = arm.string_to_tokens(base_str).to(device)
        full_tokens = arm.autoregressive_sample(base_tokens, 512-base_tokens.shape[1]).cpu()
        full_str = arm.tokens_to_string(full_tokens)
        print(f'Context: `{base_str}`')
        print(f'Continuation: `{full_str}`')
    
    def generate_ode_samples(self):
        subdir = self.sde_sample_dir()
        arm = self.construct_arm()
        arm.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        score_estimator = ScoreEstimator(arm)
        diffusion_simulator = OrnsteinUnlenbeckSimulator(score_estimator, theta=20., vocab_size=arm.config.vocab_size, max_temp=1., min_temp=0.05)
        diffusion_simulator = diffusion_simulator.to(device)
        base_str = 'My name is Donald Trump.'
        base_tokens = arm.string_to_tokens(base_str).to(device)
        simulated_tokens = diffusion_simulator.run_reverse_sde(100-base_tokens.shape[1], 1000, context=base_tokens).cpu()
        simulated_str = arm.tokens_to_string(simulated_tokens)
        print(f'Context: `{base_str}`')
        print(f'Continuation: `{simulated_str}`')
    
    def explore_loss_landscape(self):
        subdir = self.loss_landscape_dir()
        sequences_to_test: int = 8
        noise_init_per_sequence: int = 100
        arm = self.construct_arm()
        arm.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_str = 'My name is Donald Trump.'
        base_tokens = arm.string_to_tokens(base_str).to(device)
        full_tokens = arm.autoregressive_sample(base_tokens, 512-base_tokens.shape[1]).cpu()
        full_token_ll = nn.functional.cross_entropy(arm(full_tokens), full_tokens)