import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


# Load dataset
class MNIST_dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def  __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

mnist_train = datasets.MNIST(root='./data', 
                             train=True, 
                             download=True, 
                             transform=transforms.ToTensor())

mnist_test = datasets.MNIST(root='./data', 
                            train=False, 
                            download=True, 
                            transform=transforms.ToTensor())

train_images = mnist_train.data  
train_labels = mnist_train.targets  

test_images = mnist_test.data  
test_labels = mnist_test.targets  


batch_size = 256
traindataset = MNIST_dataset(train_images, train_labels)
testdataset = MNIST_dataset(test_images, test_labels)
train_loader = torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size, shuffle=True, drop_last=True)


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

# Load model
from SNN_Dropout import FCSNN

model1 = FCSNN(10)
model_path = r"nets/FCSNN_dropout.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model1.load_state_dict(state_dict)
model1.eval()

model2 = FCSNN(10)  
model_path = r"nets/FCSNN_direct_dropout.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model2.load_state_dict(state_dict)
model2.eval()

model3 = FCSNN(10) 
model_path = r"nets/FCSNN_indirect_dropout2.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model3.load_state_dict(state_dict)
model3.eval()


original = []
direct = []
indirect = []

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
    ax1.set_title('Uncertainty and Ratio for SNN-Dropout-MNISTNet', fontsize=28)    
    ax1.set_xlabel('Parameter Noise Variance', fontsize=28)
    ax1.set_ylabel('Output Variance', fontsize=28)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=28)  
    ax1.yaxis.get_offset_text().set_fontsize(28) 
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))  
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))  

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

    ax1.legend(handles, labels, loc='lower right', fontsize=28)

    os.makedirs("figs", exist_ok=True)
    plt.savefig('figs/Uncertainty and Ratio for SNN-Dropout-MNISTNet.pdf', format='pdf',bbox_inches='tight')

# Get one batch
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Choose one pic
x_0 = images[0].unsqueeze(0)  

# Call the visualization function and set the range of noise variance from 0 to 0.5 with 20 points
noise_variances = np.linspace(0, 0.5, 20)  

models = [model1, model2, model3]

x_0 = x_0.float() / 255.0  

compare_models_variance(models, x_0, noise_variances, num_samples=100)
