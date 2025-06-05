from ..generic_trial import Trial
from .config import Config
from models.autoregressive_diffusion.arm import ARM

class TextGenerationTrial(Trial):
    def __init__(self, output_dir: str, config: Config):
        super().__init__(output_dir)

        self.config = config
        self.autoregressive_sample_dir = self.register_subtrial_dir('autoregressive_samples')
        self.sde_sample_dir = self.register_subtrial_dir('sde_samples')
        self.ode_sample_dir = self.register_subtrial_dir('ode_samples')
        self.register_experiment_method('generate_autoregressive_samples', self.generate_autoregressive_samples)

    def construct_arm(self):
        arm = ARM(self.config.arm_config)
        return arm

    def generate_autoregressive_samples(self):
        subdir = self.autoregressive_sample_dir()
        arm = self.construct_arm()
        base_str = r'I pledge allegiance '
        base_tokens = arm.string_to_tokens(base_str)
        full_tokens = arm.autoregressive_sample(base_tokens, 1024-base_tokens.shape[1])
        full_str = arm.tokens_to_string(full_tokens)
        print(f'Context: `{base_str}`')
        print(f'Continuation: `{full_str}`')