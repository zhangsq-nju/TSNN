import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import math

class LinearDropConnect(nn.Module):
    def __init__(self, in_features, out_features, drop_connect_rate=0.4):
        super(LinearDropConnect, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_connect_rate = drop_connect_rate
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, drop=True):
        if drop:
            # 生成一个与权重形状相同的随机掩码，根据DropConnect rate决定哪些连接要被丢弃
            mask = torch.bernoulli(torch.ones_like(self.weight)*(1-self.drop_connect_rate))
            masked_weight = self.weight*mask
            return nn.functional.linear(x, masked_weight, self.bias)


        else:
            return nn.functional.linear(x, self.weight, self.bias)

class FCSNN(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.fc = nn.Sequential(
            LinearDropConnect(784, 500, 0.4),
            neuron.IFNode(surrogate_function=surrogate.ATan()), 
            LinearDropConnect(500, 500, 0.3),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            LinearDropConnect(500, 500, 0.2),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            LinearDropConnect(500, 10, 0.1),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = x_seq.view(x_seq.size(0), x_seq.size(1), -1)
        x_seq = self.fc(x_seq)
        fr = x_seq.mean(0)
        return fr


