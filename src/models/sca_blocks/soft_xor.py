import torch

def soft_xor(x, y):
    N = x.shape[-1]
    indices = torch.arange(N, device=x.device).unsqueeze(1) ^ torch.arange(N, device=x.device).unsqueeze(0)
    z = torch.logsumexp(x.unsqueeze(2) + y[:, indices], dim=1)
    return z