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
import torch.nn.functional as F

import zipfile

lamda = 10
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

def get_best_threshold(input, search, c=0.9):
    norm_1 = torch.norm(input, p=1)
    count = input.numel()
    threshold = norm_1 / count
    best_threshold = threshold
    # search decides whether to optimize based on regularization
    if search:
        min_loss = 1e9
        for test_thres in torch.linspace(threshold*(1-c), threshold*(1+c), 50):
            g1 = torch.sum(input<=-1*test_thres)
            g2 = torch.sum(input>=1*test_thres)
            g3 = g2 - g1
            g4 = g1 + g2 - g3**2
            loss = torch.abs(g3) + lamda*torch.abs(g4)
            if loss < min_loss:
                min_loss = loss
                best_threshold = test_thres
    
    return best_threshold


def weight_quantize(input, search):
    threshold = get_best_threshold(input, search)

    output = torch.zeros_like(input)
    output[input >= threshold] = 1
    output[input <= -1*threshold] = -1
    return output

def net_quantize(net, search):
    for layer in net.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight.data = weight_quantize(layer.weight.data, search)


def param_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.data = p1.data


def grad_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.grad = p1.grad

def param_noise(net, var):
    std = np.sqrt(var)
    for param in net.parameters():
        noise = torch.normal(0, std, param.data.size()).cuda()
        param.data += noise

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
    trainLoader = DataLoader(dataset=trainingSet, batch_size=1, shuffle=False, num_workers=4)

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
    
    for i, (inputs, targets, labels) in enumerate(trainLoader, 0):
        input = inputs.cuda()
        break

    state1 = torch.load('MLP_regularization_direct_SNN.pth')
    state2 = torch.load('MLP_direct_SNN.pth')
    n_times=100
    # record the output variance in difference noise variance
    var_rec_wo_regular = []
    var_rec_regular = []
    ratio_rec = []
    
    for var in tqdm(np.linspace(0, 0.5, 20)):
        outputs_regular = []
        for t in range(n_times):
            # reload the model in every experiment
            net.load_state_dict(state1)
            param_noise(net, var)
            param_copy(net, net_q)
            net_quantize(net_q, search=True)
            output = net_q.forward(input)
            output = torch.sum(output, dim=-1).view(1,10)
            # convert output into probability form
            output = F.softmax(output, dim=-1)
            outputs_regular.append(output)

        output_tensor = torch.cat(outputs_regular, dim=0)
        out_var = torch.var(output_tensor, dim=0, keepdim=False).mean().detach().cpu().numpy()
        var_rec_regular.append(out_var)

    for var in tqdm(np.linspace(0, 0.5, 20)):
        outputs_wo_regular = []
        for t in range(n_times):
            # reload the model in every experiment
            net.load_state_dict(state2)
            param_noise(net, var)
            param_copy(net, net_q)
            net_quantize(net_q, search=False)
            output = net_q.forward(input)
            output = torch.sum(output, dim=-1).view(1,10)
            # convert output into probability form
            output = F.softmax(output, dim=-1)
            outputs_wo_regular.append(output)

        output_tensor = torch.cat(outputs_wo_regular, dim=0)
        out_var = torch.var(output_tensor, dim=0, keepdim=False).mean().detach().cpu().numpy()
        var_rec_wo_regular.append(out_var)
    

    for i in range(len(var_rec_regular)):
        ratio = var_rec_wo_regular[i]/var_rec_regular[i]
        ratio_rec.append(ratio)
        print('var_wo_regular:{} var_regular:{} delta:{}'.format(var_rec_wo_regular[i], var_rec_regular[i], var_rec_wo_regular[i]-var_rec_regular[i]))


    fig, ax1 = plt.subplots(figsize=(10,10))


    # Plot the variance curve using the left y-axis
    colors = ['tab:red', 'tab:blue', 'tab:purple']
    ax1.set_xlabel('Parameter Noise Variance', fontsize=40)
    ax1.set_ylabel('Output Variance', fontsize=40)
    ax1.plot(np.linspace(0, 0.5, 20), var_rec_regular, color=colors[0], label='with regularization', linewidth=3)
    ax1.plot(np.linspace(0, 0.5, 20), var_rec_wo_regular, color=colors[1], label='without regularization', linewidth=3)
    # Set scientific notation
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # Increase the font size of tick labels
    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax1.yaxis.get_offset_text().set_fontsize(40)  # Adjust the font size of the exponent
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # Reduce the number of x-axis ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # Reduce the number of y-axis ticks
    # Get the legend handles and labels
    handles1, labels1 = ax1.get_legend_handles_labels()

    # Plot the ratio curve using the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ratio', fontsize=40)
    ax2.plot(np.linspace(0, 0.5, 20), ratio_rec, color=colors[2], label='Ratio', linewidth=3)
    # Increase the font size of tick labels
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax2.set_yticks([0,1,2])  
    # Get the legend handles and labels, ratio does not use scientific notation
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='lower right', fontsize=30)
    plt.savefig('Slayer_curves.pdf')


