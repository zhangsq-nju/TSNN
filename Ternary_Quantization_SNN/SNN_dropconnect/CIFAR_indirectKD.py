import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import sys
import datetime
import gzip
import math
import numpy as np
from tqdm import tqdm
import argparse

_seed_ = 42
torch.manual_seed(_seed_)
np.random.seed(_seed_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-T', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--data_dir', type=str, default='data/CIFAR10/')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_size = args.batch_size
    T = args.T
    lr = args.learning_rate
    epochs = args.epochs


    def load_cifar10_data(batch_size=batch_size):
        # define the preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # load train dataset
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2, drop_last=True)

        # load test dataset
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=True, num_workers=2, drop_last=True)

        
        return trainloader, testloader


    train_loader, test_loader = load_cifar10_data(batch_size)


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

    class CSNN(nn.Module):
        def __init__(self, T:int):
            super(CSNN, self).__init__()
            self.fc = nn.Sequential(
                layer.Conv2d(3, 20, kernel_size=5),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.AvgPool2d(kernel_size=2, stride=2),
                layer.Conv2d(20, 50, kernel_size=5),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.AvgPool2d(kernel_size=2, stride=2),
                layer.Flatten(),
                LinearDropConnect(1250, 500, 0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()), 
                LinearDropConnect(500, 500, 0.3),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                LinearDropConnect(500, 500, 0.2),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                LinearDropConnect(500, 10, 0.1),
            )

            self.layer1 = nn.Sequential(
                layer.Conv2d(3, 20, kernel_size=5),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.AvgPool2d(kernel_size=2, stride=2),
                layer.Conv2d(20, 50, kernel_size=5),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.AvgPool2d(kernel_size=2, stride=2),
                layer.Flatten(),
                LinearDropConnect(1250, 500, 0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )
            self.layer2 = nn.Sequential(
                LinearDropConnect(500, 500, 0.3),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

            self.layer3 = nn.Sequential(
                LinearDropConnect(500, 500, 0.2),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                LinearDropConnect(500, 10, 0.1),
            )
            
            self.T = T
            functional.set_step_mode(self, 'm')

        def forward(self, input):
            input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            input = self.layer1(input)
            input = self.layer2(input)
            input = self.layer3(input)

            output = input.mean(0)
            return output

        def forward_hook(self, input):
            mid_outputs = {}
            input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            input = self.layer1(input)
            mid_outputs['layer1'] = input.mean(0)
            input = self.layer2(input)
            mid_outputs['layer2'] = input.mean(0)
            input = self.layer3(input)
            mid_outputs['layer3'] = input.mean(0)

            output = input.mean(0)
            return mid_outputs, output


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


    # 跨网络复制梯度
    def grad_copy(net1, net2):
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            p2.grad = p1.grad

    def CRD_loss(mid_outputs_tea, mid_outputs_stu, temperature=0.3, alpha=0.5):
        """
        计算对比损失
        mid_outputs_tea: 教师网络中间层输出的字典
        mid_outputs_stu: 学生网络中间层输出的字典
        targets: 真实标签
        temperature: 温度参数
        alpha: 损失权重
        """
        # calculate KL-div
        kd_loss = 0
        for key in mid_outputs_tea.keys():
            tea = mid_outputs_tea[key]
            stu = mid_outputs_stu[key]

            # 教师和学生的 logits 都通过 log_softmax 转换
            tea = tea.view(tea.size(0), -1)
            stu = stu.view(stu.size(0), -1)

            # dim=-1 表示对每个样本的特征维度归一化
            tea = F.normalize(tea, dim=-1, p=2)  # L2归一化，p=2表示欧氏距离
            stu = F.normalize(stu, dim=-1, p=2)


            tea_prob = F.softmax(tea / temperature, dim=-1)
            stu_log_prob = F.log_softmax(stu / temperature, dim=-1)
            
            # 计算学生到教师的KL散度
            kd_loss += F.kl_div(stu_log_prob, tea_prob, reduction='batchmean') * (temperature ** 2)

        return alpha * kd_loss / len(mid_outputs_tea)

    def param_noise(net, sigma):
        for param in net.parameters():
            noise = torch.randn_like(param.data)
            param.data += sigma*noise


    net1 = CSNN(T).cuda()
    net2 = CSNN(T).cuda()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()


    train_acc_rates = []
    test_acc_rates = []
    steps = []
    
    
    def train(epochs:int):
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs/SNN_dropconnect_direct_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
        for epoch in range(epochs):
            acc = 0
            total = 0
            total_loss = 0
            total_crdloss = 0
            for images, labels in train_loader:
                param_copy(net1, net2)
                net_quantize(net2)
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 3, 32, 32).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()


                mid_outputs, preds = net1.forward_hook(images)
                mid_outputs_q, preds_q = net2.forward_hook(images)

                loss = loss_fn(preds_q, labels)
                crd_loss = CRD_loss(mid_outputs, mid_outputs_q)
                total_loss += loss.item()
                total_crdloss += crd_loss.item()

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=1.0)
                grad_copy(net2, net1)

                crd_loss.backward()
                # 裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=1.0)

                optimizer1.step()
                functional.reset_net(net1)
                functional.reset_net(net2)

                acc += torch.sum(torch.argmax(preds_q, dim=1) == labels).item()
                total += labels.size(0)
            train_acc_rate = acc/total
            train_acc_rates.append(train_acc_rate)
            
            acc = 0
            total = 0
            for images, labels in test_loader:
                param_copy(net1, net2)
                net_quantize(net2)
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 3, 32, 32).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds_q = net2(images)
                functional.reset_net(net1)
                functional.reset_net(net2)
                acc += torch.sum(torch.argmax(preds_q, dim=1) == labels).item()
                total += labels.size(0)
            test_acc_rate = acc/total
            test_acc_rates.append(test_acc_rate)
            if len(steps)==0:
                steps.append(0)
            else:
                steps.append(steps[-1]+1)

            total_loss = total_loss / len(train_loader)
            total_crdloss = total_crdloss / len(train_loader)
            print('Epoch:{} train_acc:{} test_acc:{} loss:{} crdloss:{}'.format(steps[-1], train_acc_rate, test_acc_rate, total_loss, total_crdloss))
            writer.add_scalar('train_acc', train_acc_rate, steps[-1])
            writer.add_scalar('test_acc', test_acc_rate, steps[-1])
            writer.add_scalar('train_loss', total_loss, steps[-1])
            writer.add_scalar('crd_loss', total_crdloss, steps[-1])

            writer.flush()

            # save model
            if (epoch+1) % 20 == 0:
                torch.save(net1.state_dict(), os.path.join(checkpoint_dir, 'SNN_dropconnect_direct_{}_epoch{}.pth'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), epoch+1)))
            
            
        writer.close()


    train(epochs)


