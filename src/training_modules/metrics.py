import torch

def get_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (torch.argmax(logits, dim=-1) == labels).to(torch.float).mean()

def get_rank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct_logits = logits.gather(1, labels.unsqueeze(1))
    rank = (logits >= correct_logits).sum(dim=1).mean()
    return rank