import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer



class CSNN(nn.Module):
    def __init__(self, T:int):
        super(CSNN, self).__init__()
        self.fc = nn.Sequential(
            layer.Conv2d(3, 20, kernel_size=5),
            layer.Dropout(p=0.4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=2, stride=2),
            layer.Conv2d(20, 50, kernel_size=5),
            layer.Dropout(p=0.4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=2, stride=2),
            layer.Flatten(),
            layer.Linear(1250, 500),
            layer.Dropout(p=0.4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(500, 10),
        )
        
        self.T = T
        functional.set_step_mode(self, 'm')

    def forward(self, input):
        input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        output = self.fc(input)
        output = output.mean(0)
        return output
