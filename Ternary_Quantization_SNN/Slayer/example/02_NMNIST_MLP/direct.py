import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import pickle as pkl
import torch.nn as nn
import wandb

import zipfile

netParams = snn.params('network.yaml')
wandb.init(
    # set the wandb project where this run will be logged
    project="Slayer_direct",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "MLP",
    "dataset": "MNIST",
    "batch_size": 8,
    "epochs": 100,
    }
)

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
 
def weight_quantize(input):
    norm_1 = torch.norm(input, p=1)
    count = input.numel()
    threshold = norm_1 / count

    output = torch.zeros_like(input)
    output[input >= threshold] = 1
    output[input <= -1*threshold] = -1
    return output

def net_quantize(net):
    for param in net.parameters():
        param.data = weight_quantize(param.data)


def param_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.data = p1.data


def grad_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.grad = p1.grad

def param_noise(net, sigma):
    for param in net.parameters():
        noise = torch.randn_like(param.data)
        param.data += sigma*noise

if __name__ == '__main__':      
    train_acc_rates = []
    test_acc_rates = []
    # Extract NMNIST samples
    with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
        for member in zip_file.namelist():
            if not os.path.exists('./' + member):
                zip_file.extract(member, './')

    # Define the cuda device to run the code on.
    device = torch.device('cuda')

    # Create network instance.
    net = Network(netParams).to(device)
    net_q = Network(netParams).to(device)

    # Create snn loss instance.
    error = snn.loss(netParams).to(device)

    # Define optimizer module.
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, amsgrad = True)
    optimizer_q = torch.optim.Adam(net_q.parameters(), lr=0.001, amsgrad=True)

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

    # Learning stats instance.
    
    # Visualize the input spikes (first five samples).
    for i in range(5):
        input, target, label = trainingSet[i]
        snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))
    
    net.load_state_dict(torch.load('MLP_SNN.pth'))
    # Main loop
    def train(epochs:int):
        for epoch in tqdm(range(epochs)):
            tSt = datetime.now()
            acc = 0
            total = 0
            for i, (input, target, label) in enumerate(trainLoader, 0):
                param_copy(net, net_q)
                net_quantize(net_q)
                # Move the input and target to correct GPU.
                input  = input.to(device)
                target = target.to(device) 
                
                # Forward pass of the network.
                output = net_q.forward(input)
                
                # Gather the training stats.
                acc += torch.sum( snn.predict.getClass(output) == label ).data.item()
                total += len(label)
                # Calculate loss.
                loss = error.numSpikes(output, target)
                
                # Reset gradients to zero.
                
                optimizer.zero_grad()
                optimizer_q.zero_grad()
                # Backward pass of the network.
                loss.backward()
                grad_copy(net_q, net)
                # Update weights.
                optimizer.step()


            train_acc_rate = acc/total
            train_acc_rates.append(train_acc_rate)
            # Testing loop.
            # Same steps as Training loops except loss backpropagation and weight update.
            acc = 0
            total = 0
            for i, (input, target, label) in enumerate(testLoader, 0):
                input  = input.to(device)
                target = target.to(device) 
                
                output = net_q.forward(input)

                acc += torch.sum( snn.predict.getClass(output) == label ).data.item()
                total += len(label)
                loss = error.numSpikes(output, target)
            test_acc_rate = acc/total
            test_acc_rates.append(test_acc_rate)
            wandb.log({'direct_train_acc_rate': train_acc_rate, 'direct_test_acc_rate': test_acc_rate})
    
    
    def test(epoch):
        acc = 0
        total = 0
        for i, (input, target, label) in enumerate(testLoader, 0):
            input  = input.to(device)
            target = target.to(device) 
            
            output = net_q.forward(input)

            acc += torch.sum( snn.predict.getClass(output) == label ).data.item()
            total += len(label)
            loss = error.numSpikes(output, target)
        acc_rate = acc/total
        train_acc_rates.append(acc_rate)
        test_acc_rates.append(acc_rate)

    train(100)

    wandb.finish()
