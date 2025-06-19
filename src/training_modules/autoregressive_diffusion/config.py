from dataclasses import dataclass

@dataclass
class DiffusionConfig: # based on the paper 'Elucidating the Design Space of Diffusion Models' (Karras et al. 2022)
    sigma_min: float = 0.002
    sigma_max: float = 80.
    sigma_data: float = 0.5
    rho: float = 7.
    p_mean: float = -1.2
    p_std: float = 1.2
    s_churn: float = 40.
    s_tmin: float = 0.05
    s_tmax: float = 50.
    s_noise: float = 1.003

    def __post_init__(self):
        assert isinstance(self.sigma_min, float) and (0 < self.sigma_min < float('inf'))
        assert isinstance(self.sigma_max, float) and (self.sigma_min < self.sigma_max < float('inf'))
        assert isinstance(self.sigma_data, float) and (0 < self.sigma_data < float('inf'))
        assert isinstance(self.rho, float) and (0 < self.rho < float('inf'))
        assert isinstance(self.p_mean, float) and (-float('inf') < self.p_mean < float('inf'))
        assert isinstance(self.p_std, float) and (0 < self.p_std < float('inf'))