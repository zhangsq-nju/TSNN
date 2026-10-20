import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import copy
import os

from SNN_slayer import Network
import slayerSNN as snn

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
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]
    
device = torch.device('cuda')
netParams = snn.params('network.yaml')

# Create network instance.
net = Network(netParams).to(device)

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



# Load model
# Pass the configuration to the constructor of the Network class
model1 = Network(netParams)
model_path = r"nets/MLP_SNN.pth"
state_dict = torch.load(model_path, map_location=device)
model1.load_state_dict(state_dict)
model1.to(device)
model1.eval()

model2 = Network(netParams)
model_path = r"nets/MLP_direct_SNN.pth"
state_dict = torch.load(model_path, map_location=device)
model2.load_state_dict(state_dict)
model2.to(device)
model2.eval()

model3 = Network(netParams)
model_path = r"nets/MLP_indirect_SNN2.pth"
state_dict = torch.load(model_path, map_location=device)
model3.load_state_dict(state_dict)
model3.to(device)
model3.eval()


original = []
direct = []
indirect = []

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

def compute_output_variance(model, x_0, noise_variance, num_samples=100):
    output_samples = []
    
    for _ in range(num_samples):
        model_copy = copy.deepcopy(model)
        
        # Add noise to the parameters of each layer
        for param in model_copy.parameters():
            if param.requires_grad:  
                param_data = param.data
                noise_std = np.sqrt(noise_variance)
                noise = torch.normal(0, noise_std, size=param_data.shape).to(param.device)
                param_data += noise      
        
        if id(model)==id(model2) or id(model)==id(model3):
            net_quantize(model_copy)
        
        # Calculate the output of the current model
        with torch.no_grad():
            output = model_copy(x_0) 
            output_samples.append(output)
    
    # Combine the outputs of all samples together
    all_outputs = torch.cat(output_samples, dim=0)

    # Apply Softmax along the dimension dim = 1
    normalized_outputs = torch.softmax(all_outputs, dim=1)

    output_variance = normalized_outputs.var(dim=0)
    return output_variance.mean().item()

def plot_variance_vs_output(model, x_0, noise_variances, num_samples=100, label=None, color=None):
    global original, direct, indirect 
    output_variances = []
    
    for noise_variance in noise_variances:
        output_variance = compute_output_variance(model, x_0, noise_variance, num_samples)
        output_variances.append(output_variance)
    
    # Normalization
    output_variances=np.array(output_variances)
    output_sum=np.sum(output_variances)
    output_variances=output_variances/output_sum
    if id(model)==id(model1):original = copy.deepcopy(output_variances)
    if id(model)==id(model2):direct = copy.deepcopy(output_variances)
    if id(model)==id(model3):indirect = copy.deepcopy(output_variances)

    plt.plot(noise_variances, output_variances, label=label, color=color)

def compare_models_variance(models, x_0, noise_variances, num_samples=100):
    # Plot the curves of each model using different colors and labels
    colors = ['r', 'g', 'b']
    labels = ['FP', 'Direct', 'Indirect']
    
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # Draw the first graph (using the vertical axis on the left side)
    for i, model in enumerate(models):
        plot_variance_vs_output(model, x_0, noise_variances, num_samples, label=labels[i], color=colors[i])
    ax1.set_title('Uncertainty and Ratio for SNN-Slayer-MNISTNet', fontsize=28)
    ax1.set_xlabel('Parameter Noise Variance', fontsize=28)
    ax1.set_ylabel('Output Variance', fontsize=28)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) 
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=28)  
    ax1.yaxis.get_offset_text().set_fontsize(28) 
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))  
    ax1.set_yticks([0,0.04,0.08])

    # Draw the second graph (using the vertical axis on the right side)
    ax2 = ax1.twinx()
    epsilon = 1e-4
    ratio_variances = [o / (d + epsilon) for o, d in zip(original, direct)]
    
    ax2.plot(noise_variances, ratio_variances, label='Ratio', color='purple')
    ax2.set_ylabel('Ratio', fontsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)  # 放大刻度标签字体大小
    ax2.set_yticks([0,1,2])
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    handles = handles1 + handles2
    labels = labels1 + labels2

    ax1.legend(handles, labels, loc='lower right', fontsize=30)

    os.makedirs("figs", exist_ok=True)
    plt.savefig('figs/Uncertainty and Ratio for SNN-Slayer-MNISTNet.pdf', format='pdf',bbox_inches='tight')

# Fetch data from the testLoader
data_iter = iter(testLoader)
inputs, labels, classLabels = next(data_iter)
inputs = inputs.to(device)
labels = labels.to(device)

# Get one pic（inputSpikes）
inputSpikes = inputs[0]  

# `inputSpikes` is a 4D tensor with the shape of (channels, height, width, time)
# Select the last time step
time_step = -1  
image = inputSpikes[0, :, :, time_step].cpu().numpy()  
image = image / np.max(image)  

# Call the visualization function and set the range of noise variance from 0 to 0.5 with 20 points
noise_variances = np.linspace(0, 0.5, 20)

models = [model1, model2, model3]
compare_models_variance(models, inputs, noise_variances, num_samples=100)


