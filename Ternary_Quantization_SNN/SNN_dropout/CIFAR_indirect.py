import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import Dataset, DataLoader
import os
import time
import sys
import datetime
import gzip
import wandb
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-T', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=99)
    parser.add_argument('--qn', type=int, default=100)

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    T = args.T
    lr = args.learning_rate
    epochs = args.epochs
    qn = args.qn


    def load_cifar10_data(batch_size=64):
        # define the preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # load train datasets
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2, drop_last=True)

        # load test datasets
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2, drop_last=True)

        
        return trainloader, testloader


    train_loader, test_loader = load_cifar10_data(batch_size)

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

    def weight_quantize(input):
        norm_1 = torch.norm(input, p=1)
        count = input.numel()
        threshold = norm_1 / count

        output = torch.zeros_like(input)
        output[input >= threshold] = 1
        output[input <= -1*threshold] = -1
        return output

    def net_quantize(net):
        for layer in net.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.weight.data = weight_quantize(layer.weight.data)


    def param_copy(net1, net2):
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            p2.data = p1.data


    def grad_copy(net1, net2):
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            p2.grad = p1.grad


    wandb.init(
        dir=args.output_dir,
    # set the wandb project where this run will be logged
    project="SNN_dropout_indirect",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "SNN_dropout",
    "epochs": epochs,
    }
    )

    net1 = CSNN(T).cuda()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()


    train_acc_rates = []
    test_acc_rates = []
    steps = []
        


    def train(epochs:int):
        for epoch in range(epochs):
            acc = 0
            total = 0
            for images, labels in train_loader:
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 3, 32, 32).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds = net1(images)
                loss = loss_fn(preds, labels)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                functional.reset_net(net1)
                acc += torch.sum(torch.argmax(preds, dim=1) == labels).item()
                total += labels.size(0)
            train_acc_rate = acc/total
            train_acc_rates.append(train_acc_rate)

            acc = 0
            total = 0
            for images, labels in test_loader:
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 3, 32, 32).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds = net1(images)
                functional.reset_net(net1)
                acc += torch.sum(torch.argmax(preds, dim=1) == labels).item()
                total += labels.size(0)
            test_acc_rate = acc/total
            test_acc_rates.append(test_acc_rate)
            if len(steps)==0:
                steps.append(0)
            else:
                steps.append(steps[-1]+1)

            print('Epoch:{} train_acc:{} test_acc:{}'.format(steps[-1], train_acc_rate, test_acc_rate))
            wandb.log({'train_acc': train_acc_rate, 'test_acc': test_acc_rate})

    def test():
        acc = 0
        total = 0
        for images, labels in test_loader:
            images = torch.tensor(images, dtype=torch.float).view(batch_size, 3, 32, 32).cuda()
            labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
            preds = net1(images)
            functional.reset_net(net1)
            acc += torch.sum(torch.argmax(preds, dim=1) == labels).item()
            total += labels.size(0)
        test_acc_rate = acc/total
        test_acc_rates.append(test_acc_rate)
        if len(steps)==0:
            steps.append(0)
        else:
            steps.append(steps[-1]+1)

        wandb.log({'test_acc': test_acc_rate})

    for i in range(qn):
        train(epochs)
        net_quantize(net1)
        test()

    wandb.finish()


