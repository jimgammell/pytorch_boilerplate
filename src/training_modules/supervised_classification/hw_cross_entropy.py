import numpy as np
import torch
from torch import nn

class HWSmoothedCrossEntropy(nn.Module):
    byte_to_dist_lut: torch.Tensor

    def __init__(self, label_smoothing: float = 0.0, **kwargs):
        super().__init__()
        self.label_smoothing = label_smoothing
        byte_to_hw_lut = np.zeros((256,), dtype=np.uint8)
        for byte in np.arange(256, dtype=np.uint8):
            hw = np.unpackbits(byte).sum()
            byte_to_hw_lut[byte] = hw
        byte_to_dist_lut = np.zeros((256, 256), dtype=np.float32)
        for byte1 in np.arange(256, dtype=np.uint8):
            hw1 = byte_to_hw_lut[byte1]
            for byte2 in np.arange(256, dtype=np.uint8):
                hw2 = byte_to_hw_lut[byte2]
                byte_to_dist_lut[byte1, byte2] = np.float32(hw1 == hw2)
        byte_to_dist_lut /= byte_to_dist_lut.sum(axis=1)
        self.register_buffer('byte_to_dist_lut', torch.from_numpy(byte_to_dist_lut))
        self.cross_entropy = nn.CrossEntropyLoss(**kwargs)
    
    def forward(self, logits, labels):
        label_true_dist = nn.functional.one_hot(labels, num_classes=256).to(logits.dtype)
        label_hw_dist = self.byte_to_dist_lut[labels, :]
        label_dist = (1.-self.label_smoothing)*label_true_dist + self.label_smoothing*label_hw_dist
        loss = self.cross_entropy(logits, label_dist)
        return loss