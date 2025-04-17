from typing import List, Tuple

import torch
from torch import nn

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.construct()
        self.init_weights()
    
    def construct(self):
        pass

    def init_weights(self):
        pass

    def extra_repr(self):
        return f'Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}'

    def get_params_based_on_should_weight_decay(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        raise NotImplementedError