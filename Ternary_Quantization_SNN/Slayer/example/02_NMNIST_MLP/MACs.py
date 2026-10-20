from ptflops import get_model_complexity_info
import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")
import torch.nn as nn
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
import zipfile

netParams = snn.params('network.yaml')

# Dataset definition
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]

        inputSpikes = snn.io.read2Dspikes(
                        self.path + str(inputIndex.item()) + '.bs2'
                        ).toSpikeTensor(torch.zeros((2,34,34,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel,...] = 1
        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]

# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # Initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Define network functions
        # The commented line below should be used if the input spikes were not reshaped
        # self.fc1   = slayer.dense((34, 34, 2), 512)
        self.fc1   = slayer.dense((34*34*2), 512)
        self.fc2   = slayer.dense(512, 10)

    def forward(self, spikeInput):
        # Both set of definitions are equivalent. The uncommented version is much faster.
        
        # spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(spikeInput)))
        # spikeLayer2 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1)))

        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))     
        
        return spikeLayer2
        # return spikeInput, spikeLayer1, spikeLayer2
 

# Extract NMNIST samples
with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
    for member in zip_file.namelist():
        if not os.path.exists('./' + member):
            zip_file.extract(member, './')

# Define the cuda device to run the code on.
device = torch.device('cuda')

# Create network instance.
net = Network(netParams).to(device)

# Create snn loss instance.
error = snn.loss(netParams).to(device)

# Define optimizer module.
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

# Dataset and dataLoader instances.
trainingSet = nmnistDataset(datasetPath =netParams['training']['path']['in'], 
                            sampleFile  =netParams['training']['path']['train'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'], 
                            sampleFile  =netParams['training']['path']['test'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=4)


class Simple_SLayer(nn.Module):
    def __init__(self):
        super(Simple_SLayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(2312, 512),
            nn.Linear(512, 10),
        )


    def forward(self, x):
        x = self.fc(x)
        return x

model = Simple_SLayer()
input_shape = (1, 2312)
macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True, verbose=True)
print("MACs:", macs)
print("Params:", params)
