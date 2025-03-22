import torch

def get_accuracy(logits: torch.Tensor, labels: torch.Tensor, avg_result: bool = True) -> torch.Tensor:
    rv = (torch.argmax(logits, dim=-1) == labels).to(torch.float)
    if avg_result:
        rv = rv.mean()
    return rv

def get_rank(logits: torch.Tensor, labels: torch.Tensor, avg_result: bool = True) -> torch.Tensor:
    correct_logits = logits.gather(1, labels.unsqueeze(1))
    rank = (logits >= correct_logits).to(torch.float).sum(dim=1)
    if avg_result:
        rank = rank.mean()
    return rank