import torch
import torch.nn as nn
import numpy as np
from typing import Literal

class LoRA(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank=4, alpha=1.0, scaling_type: Literal["r/a", "r/sqrta"] = "r/a"):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        if scaling_type == "r/a":
            self.scaling = alpha / rank
        elif scaling_type == "r/sqrta":
            self.scaling = alpha / np.sqrt(rank)
        self.lora_A=nn.Parameter(torch.empty((rank, original_layer.in_features)))
        self.lora_B=nn.Parameter(torch.zeros((original_layer.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A)
        
    def forward(self, x):
        return self.original_layer(x) + (self.lora_B @ self.lora_A @ x.T).T * self.scaling
    

def inject_lora(model: nn.Module, rank=4, alpha=1.0, scaling_type: Literal["r/a", "r/sqrta"] = "r/a") -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRA(module, rank=rank, alpha=alpha, scaling_type=scaling_type))
        else:
            inject_lora(module, rank=rank, alpha=alpha, scaling_type=scaling_type)
    return model
