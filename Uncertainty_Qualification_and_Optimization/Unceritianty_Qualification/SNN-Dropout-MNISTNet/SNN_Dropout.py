import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class FCSNN(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.fc = nn.Sequential(
            layer.Linear(784, 500),
            nn.Dropout(p=0.4),
            neuron.IFNode(surrogate_function=surrogate.ATan()), 
            layer.Linear(500, 500),
            nn.Dropout(p=0.4),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Dropout(p=0.4),
            layer.Linear(500, 500),
            nn.Dropout(p=0.4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(500, 10),
        )
        
        functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = x_seq.view(x_seq.size(0), x_seq.size(1), -1)
        x_seq = self.fc(x_seq)
        fr = x_seq.mean(0)
        return fr
