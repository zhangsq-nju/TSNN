import torch
from torch.utils.data import Dataset, DataLoader
import src.slayerSNN as snn


class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # Initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Define network functions
        self.fc1   = slayer.dense((34*34*2), 512)
        self.fc2   = slayer.dense(512, 10)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))     
        
        return spikeLayer2
        
 