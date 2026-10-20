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
import numpy as np
from tqdm import tqdm
import argparse

_seed_ = 42
torch.manual_seed(_seed_)
np.random.seed(_seed_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('-T', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')

    parser.add_argument('--data_dir', type=str, default='data/MNIST')
    parser.add_argument('--output_dir', type=str, default='.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(args)
    batch_size = args.batch_size
    T = args.T
    lr = args.learning_rate
    epochs = args.epochs

    
    class MNIST_dataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        
        def __len__(self):
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


    class FCSNN(nn.Module):
        def __init__(self, T: int):
            super().__init__()
            self.T = T
            self.fc = nn.Sequential(
                layer.Linear(784, 500),
                nn.Dropout(p=0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()), 
                layer.Linear(500, 500),
                nn.Dropout(p=0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.Dropout(p=0.4),
                layer.Linear(500, 500),
                nn.Dropout(p=0.4),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.Linear(500, 10),
            )

            self.layer1 = nn.Sequential(
                layer.Linear(784, 500),
                nn.Dropout(p=0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

            self.layer2 = nn.Sequential(
                layer.Linear(500, 500),
                nn.Dropout(p=0.4),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

            self.layer3 = nn.Sequential(
                nn.Dropout(p=0.4),
                layer.Linear(500, 500),
                nn.Dropout(p=0.4),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                layer.Linear(500, 10),
            )
            functional.set_step_mode(self, 'm')

        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
            x_seq = x_seq.view(x_seq.size(0), x_seq.size(1), -1)
            input = self.layer1(x_seq)
            input = self.layer2(input)
            input = self.layer3(input)
            fr = input.mean(0)
            return fr

        def forward_hook(self, input):
            mid_outputs = {}
            input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            input = input.view(input.size(0), input.size(1), -1)
            input = self.layer1(input)
            mid_outputs['layer1'] = input
            input = self.layer2(input)
            mid_outputs['layer2'] = input
            input = self.layer3(input)
            mid_outputs['layer3'] = input

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
            if p2.grad is not None:
                p2.grad += p1.grad
            else:
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


    net1 = FCSNN(T).cuda()
    net2 = FCSNN(T).cuda()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()


    train_acc_rates = []
    test_acc_rates = []
    steps = []
    
    # Variable to track the best test accuracy
    best_test_acc = 0.0

    def train(epochs:int):
        best_test_acc = 0.0
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs/SNN_dropout_direct_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
        for epoch in range(epochs): 
            acc = 0
            total = 0
            total_loss = 0
            total_crdloss = 0
            for images, labels in train_loader:
                param_copy(net1, net2)
                net_quantize(net2)
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 1, 28, 28).cuda()
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
                images = torch.tensor(images, dtype=torch.float).view(batch_size, 1, 28, 28).cuda()
                labels = torch.tensor(labels, dtype=torch.long).view(-1).cuda()
                preds_q = net2(images)
                functional.reset_net(net1)
                functional.reset_net(net2)
                acc += torch.sum(torch.argmax(preds_q, dim=1) == labels).item()
                total += labels.size(0)
            test_acc_rate = acc/total
            test_acc_rates.append(test_acc_rate)

            # Check if current test accuracy is the best so far
            if test_acc_rate > best_test_acc:
                best_test_acc = test_acc_rate
                # Save the model state
                checkpoint_path = os.path.join(checkpoint_dir, f'model_best.pth')
                torch.save(net1.state_dict(), checkpoint_path)
                print(f"Saved new best model with test accuracy: {best_test_acc:.4f}")

            # Save a checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(net1.state_dict(), checkpoint_path)
                print(f"Saved checkpoint for epoch {epoch+1}")
                

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
            
        
        writer.close()


    train(epochs)