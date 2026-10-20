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
from tqdm import tqdm
import math
import argparse
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('-T', type=int, default=10)

    parser.add_argument('--data_dir', type=str, default='MNIST')
    parser.add_argument('--output_dir', type=str, default='.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    T = args.T

    run = wandb.init(
        dir=args.output_dir,
    # Set the wandb project where this run will be logged.
    project="SNN_dropconnect_direct",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "SNN_dropconnect",
        "epochs": epochs,
        "batch_size": batch_size
    },
    )


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
            # read the first 16 bytes
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
            # read the first 8 bytes
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

        def forward(self, x):
            # Generate a random mask with the same shape as the weights, 
            # and determine which connections to drop based on the DropConnect rate.
            mask = torch.bernoulli(torch.ones_like(self.weight)*(1-self.drop_connect_rate))
            masked_weight = self.weight*mask
            return nn.functional.linear(x, masked_weight, self.bias)


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

    net1 = FCSNN(T).cuda()
    net2 = FCSNN(T).cuda()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()


    train_acc_rates = []
    test_acc_rates = []
    
    
    
    def train(epochs:int):
        for epoch in range(epochs):
            acc = 0
            total = 0
            for images, labels in tqdm(train_loader):
                param_copy(net1, net2)
                net_quantize(net2)
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 1, 28, 28).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds = net2(images)
                loss = loss_fn(preds, labels)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                grad_copy(net2, net1)
                optimizer1.step()
                functional.reset_net(net1)
                functional.reset_net(net2)
                acc += torch.sum(torch.argmax(preds, dim=1) == labels).item()
                total += labels.size(0)
            train_acc_rate = acc/total
            train_acc_rates.append(train_acc_rate)

            acc = 0
            total = 0
            for images, labels in tqdm(test_loader):
                param_copy(net1, net2)
                net_quantize(net2)
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 1, 28, 28).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds = net2(images)
                functional.reset_net(net1)
                functional.reset_net(net2)
                acc += torch.sum(torch.argmax(preds, dim=1) == labels).item()
                total += labels.size(0)
            test_acc_rate = acc/total
            test_acc_rates.append(test_acc_rate)
            wandb.log({'train_acc': train_acc_rate, 'test_acc': test_acc_rate})

    train(100)

    wandb.finish()

    