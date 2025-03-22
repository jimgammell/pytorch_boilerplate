import torch
from torch import nn

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        true_prob = (-ce_loss).exp()
        focal_factor = (1 - true_prob)**self.gamma
        focal_loss = (self.alpha*focal_factor*ce_loss).mean()
        return focal_loss