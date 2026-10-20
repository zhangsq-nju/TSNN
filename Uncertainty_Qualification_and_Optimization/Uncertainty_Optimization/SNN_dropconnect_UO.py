import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import Dataset, DataLoader
import os
import time
import sys
import datetime
import gzip
import numpy as np
import wandb
from tqdm import tqdm
import math
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lamda', type=float, default=10)
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='MNIST')
    parser.add_argument('--output_dir', type=str, default='.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    lamda = args.lamda
    plot = args.plot
    path = args.model_path


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

    def load_mnist_images(filename):
        # open gzip file
        with gzip.open(filename, 'rb') as f:
            # read first 16 bytes
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_images = int.from_bytes(f.read(4), byteorder='big')
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')

            # read images
            image_data = np.frombuffer(f.read(), dtype=np.uint8)

            # reshape images
            images = image_data.reshape(num_images, num_rows, num_cols)

        return images


    def load_mnist_labels(filename):
        # open gzip file
        with gzip.open(filename, 'rb') as f:
            # read first 8 bytes
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_labels = int.from_bytes(f.read(4), byteorder='big')

            # read all labels
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels


    train_labels = load_mnist_labels(os.path.join(args.data_dir, 'train-labels-idx1-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(args.data_dir, 't10k-labels-idx1-ubyte.gz'))
    train_images = load_mnist_images(os.path.join(args.data_dir, 'train-images-idx3-ubyte.gz'))
    test_images = load_mnist_images(os.path.join(args.data_dir, 't10k-images-idx3-ubyte.gz'))


    traindataset = MNIST_dataset(train_images, train_labels)
    testdataset = MNIST_dataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size, shuffle=True, drop_last=True)


    class LinearDropConnect(nn.Module):
        def __init__(self, in_features, out_features, drop=False, drop_connect_rate=0.4):
            super(LinearDropConnect, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.drop = drop
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

        def forward(self, x):
            if self.drop:
                # Generate a random mask with the same shape as the weights, 
                # and determine which connections to drop based on the DropConnect rate.
                mask = torch.bernoulli(torch.ones_like(self.weight)*(1-self.drop_connect_rate))
                masked_weight = self.weight*mask
                return nn.functional.linear(x, masked_weight, self.bias)
            else:
                return nn.functional.linear(x, self.weight, self.bias)

    class FCSNN(nn.Module):
        def __init__(self, T: int, drop=False):
            super().__init__()
            self.T = T
            self.fc = nn.Sequential(
                LinearDropConnect(784, 500, drop, 0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()), 
                LinearDropConnect(500, 500, drop, 0.3),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                LinearDropConnect(500, 500, drop, 0.2),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                LinearDropConnect(500, 10, drop, 0.1),
            )

            functional.set_step_mode(self, 'm')

        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
            x_seq = x_seq.view(x_seq.size(0), x_seq.size(1), -1)
            x_seq = self.fc(x_seq)
            fr = x_seq.mean(0)
            return fr

    def get_best_threshold(input, search=True, c=0.9):
        norm_1 = torch.norm(input, p=1)
        count = input.numel()
        threshold = norm_1 / count
        best_threshold = threshold
        # search determines whether to perform uncertainty optimization
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


    input = train_images[0]
    input = torch.tensor(input, dtype=torch.float).view(1,1,28,28).cuda()

    
    n_times = 100

    # record the output variance in difference noise variance
    var_rec_wo_regular = []
    var_rec_regular = []
    ratio_rec = []

    for var in tqdm(np.linspace(0, 0.25, 20)):
        outputs_regular = []
        for t in range(n_times):
            # reload the model in every experiment
            net1 = torch.load(path)
            net2 = FCSNN(10).cuda()
            param_noise(net1, var)
            param_copy(net1, net2)
            net_quantize(net2, search=True)
            output = net2(input)[:,:10].view(1,10)
            # convert output into probability form
            output = F.softmax(output, dim=-1)
            outputs_regular.append(output)
            # reset net
            functional.reset_net(net1)
            functional.reset_net(net2)
        output_tensor = torch.cat(outputs_regular, dim=0)
        out_var = torch.var(output_tensor, dim=0, keepdim=False).mean().detach().cpu().numpy()
        var_rec_regular.append(out_var)

    for var in tqdm(np.linspace(0, 0.5, 20)):
        outputs_wo_regular = []
        for t in range(n_times):
            # reload the model in every experiment
            net1 = torch.load(path)
            net2 = FCSNN(10).cuda()
            param_noise(net1, var)
            param_copy(net1, net2)
            net_quantize(net2, search=False)
            output = net2(input)[:,:10].view(1,10)
            # convert output into probability form
            output = F.softmax(output, dim=-1)
            outputs_wo_regular.append(output)
            # reset net
            functional.reset_net(net1)
            functional.reset_net(net2)
        output_tensor = torch.cat(outputs_wo_regular, dim=0)
        out_var = torch.var(output_tensor, dim=0, keepdim=False).mean().detach().cpu().numpy()
        var_rec_wo_regular.append(out_var)


    for i in range(len(var_rec_regular)):
        ratio = var_rec_wo_regular[i]/var_rec_regular[i]
        ratio_rec.append(ratio)
        print('var_wo_regular:{} var_regular:{} delta:{}'.format(var_rec_wo_regular[i], var_rec_regular[i], var_rec_wo_regular[i]-var_rec_regular[i]))


    if plot:
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
        ax1.yaxis.get_offset_text().set_fontsize(40) 
        ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=3)) 
        ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))  
        # get legends
        handles1, labels1 = ax1.get_legend_handles_labels()

        # Plot the ratio curve using the right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Ratio', fontsize=40)
        ax2.plot(np.linspace(0, 0.5, 20), ratio_rec, color=colors[2], label='Ratio', linewidth=3)
        # Increase the font size of tick labels
        ax2.tick_params(axis='both', which='major', labelsize=40)
        ax2.set_yticks([0,1,2])  
        handles2, labels2 = ax2.get_legend_handles_labels()

        handles = handles1 + handles2
        labels = labels1 + labels2
        ax1.legend(handles, labels, loc='lower right', fontsize=30)

        plt.savefig(os.path.join(args.output_dir, 'SNN_dropconnect_curves.pdf'))

