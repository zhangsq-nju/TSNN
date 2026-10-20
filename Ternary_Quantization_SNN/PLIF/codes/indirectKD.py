import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import readline
import os
import argparse
import time
import sys


torch.backends.cudnn.benchmark = True

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import models
from tqdm import tqdm


def weight_quantize(input):
    norm_1 = torch.norm(input, p=1)
    count = input.numel()
    threshold = norm_1 / count

    output = torch.zeros_like(input)
    output[input >= threshold] = 1
    output[input <= -1*threshold] = -1
    return output

def net_quantize(net):
    if portion == 'all':
        for layer in net.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.weight.data = weight_quantize(layer.weight.data)
    elif portion == 'fc':
        for layer in net.modules():
            if isinstance(layer, (nn.Linear)):
                layer.weight.data = weight_quantize(layer.weight.data)
    elif portion == 'conv':
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d)):
                layer.weight.data = weight_quantize(layer.weight.data)


def param_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.data = p1.data


def grad_copy(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p2.grad = p1.grad

def CRD_loss(mid_outputs_tea, mid_outputs_stu, temperature=0.3, alpha=0.01):
    """
    计算对比损失
    mid_outputs_tea: 教师网络中间层输出的字典
    mid_outputs_stu: 学生网络中间层输出的字典
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('-init_tau', type=float, required=True)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-learning_rate', type=float, default=1e-3)
    parser.add_argument('-T_max', type=int, default=64)
    parser.add_argument('-use_plif', action='store_true', default=False)
    parser.add_argument('-alpha_learnable', action='store_true', default=False)
    parser.add_argument('-use_max_pool', action='store_true', default=False)
    parser.add_argument('-device', type=str, required=True)
    parser.add_argument('-dataset_name', type=str, required=True)
    parser.add_argument('-dataset_dir', type=str, default='./data')
    parser.add_argument('-log_dir_prefix', type=str, default='./logs')
    parser.add_argument('-running_data', type=str)
    parser.add_argument('-T', type=int, default=5)
    parser.add_argument('-channels', type=int)
    parser.add_argument('-number_layer', type=int)
    parser.add_argument('-split_by', type=str)
    parser.add_argument('-normalization', type=str)
    parser.add_argument('-max_epoch', type=int, required=True)
    parser.add_argument('-detach_reset', action='store_true', default=False)
    parser.add_argument('-portion', type=str, default='all')
    parser.add_argument('-quantize', action='store_true', default=False)
    parser.add_argument('-KD', action='store_true', default=False)

    args = parser.parse_args()
    argv = ' '.join(sys.argv)

    print(args) 
    init_tau = args.init_tau
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    T_max = args.T_max
    use_plif = args.use_plif
    alpha_learnable = args.alpha_learnable
    use_max_pool = args.use_max_pool
    device = args.device
    dataset_name = args.dataset_name
    dataset_dir = os.path.join(args.dataset_dir, dataset_name)
    log_dir_prefix = args.log_dir_prefix
    running_data = args.running_data or os.path.join(log_dir_prefix, 'running_data')
    T = args.T
    max_epoch = args.max_epoch
    detach_reset = args.detach_reset
    KD = args.KD
    quantize = args.quantize
    portion = args.portion

    number_layer = args.number_layer
    channels = args.channels
    split_by = args.split_by
    normalization = args.normalization


    if normalization == 'None':
        normalization = None

    if dataset_name != 'MNIST' and dataset_name != 'FashionMNIST' and dataset_name != 'CIFAR10':
        dir_name = f'{dataset_name}_quantize_{quantize}_KD_{KD}_portion_{portion}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_c_{channels}_n_{number_layer}_split_by_{split_by}_normalization_{normalization}_detach_reset_{detach_reset}'
    else:
        dir_name = f'{dataset_name}_quantize_{quantize}_KD_{KD}_portion_{portion}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_detach_reset_{detach_reset}_time_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    log_dir = os.path.join(log_dir_prefix, dir_name)
    running_data_folder = os.path.join(running_data, dir_name)
    os.makedirs(running_data_folder, exist_ok=True)
    running_data_file = os.path.join(running_data_folder, 'running_data.txt')
    pt_dir = os.path.join(log_dir_prefix, 'plif_pt_' + dir_name)
    pt_dir2 = os.path.join(log_dir_prefix, 'cifar_plif_pt_' + dir_name)
    print(log_dir, pt_dir)
    os.makedirs(pt_dir, exist_ok=True)
    class_num = 10  # 绝大多数数据集为10，如果不一样，在新建dataloader的时候再修改
    if dataset_name == 'MNIST':
        transform_train, transform_test = models.get_transforms(dataset_name)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 8,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'FashionMNIST':
        transform_train, transform_test = models.get_transforms(dataset_name)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'CIFAR10':
        transform_train, transform_test = models.get_transforms(dataset_name)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'NMNIST':
        from spikingjelly.datasets.n_mnist import NMNIST

        train_data_loader = torch.utils.data.DataLoader(
            dataset=NMNIST(dataset_dir, train=True, use_frame=True, frames_num=T, split_by=split_by,
                           normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=NMNIST(dataset_dir, train=False, use_frame=True, frames_num=T, split_by=split_by,
                           normalization=normalization),
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

        train_data_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10DVS(dataset_dir, train=True, split_ratio=0.9, use_frame=True, frames_num=T,
                               split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10DVS(dataset_dir, train=False, split_ratio=0.9, use_frame=True, frames_num=T,
                               split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'ASLDVS':
        class_num = 24
        from spikingjelly.datasets.asl_dvs import ASLDVS

        train_data_loader = torch.utils.data.DataLoader(
            dataset=ASLDVS(dataset_dir, train=True, split_ratio=0.8, use_frame=True, frames_num=T,
                           split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=ASLDVS(dataset_dir, train=False, split_ratio=0.8, use_frame=True, frames_num=T,
                           split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'DVS128Gesture':
        class_num = 11
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

        train_data_loader = torch.utils.data.DataLoader(
            dataset=DVS128Gesture(dataset_dir, train=True, use_frame=True, frames_num=T,
                                  split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=DVS128Gesture(dataset_dir, train=False, use_frame=True, frames_num=T,
                                  split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)

    check_point_path = os.path.join(pt_dir, 'check_point.pt')
    check_point_max_path = os.path.join(pt_dir, 'check_point_max.pt')
    check_point_max_path2 = os.path.join(pt_dir2, 'check_point_max.pt')

    net_max_path = os.path.join(pt_dir, 'net_max.pt')
    optimizer_max_path = os.path.join(pt_dir, 'optimizer_max.pt')
    scheduler_max_path = os.path.join(pt_dir, 'scheduler_max.pt')
    check_point = None
    
    if dataset_name == 'MNIST':
        net1 = models.MNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        net2 = models.MNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
    elif dataset_name == 'FashionMNIST':
        net1 = models.FashionMNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                        alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        net2 = models.FashionMNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                        alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
    elif dataset_name == 'CIFAR10':
        net1 = models.Cifar10Net(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        net2 = models.Cifar10Net(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
    elif dataset_name == 'NMNIST':
        net = models.NMNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                number_layer=number_layer).to(device)
    elif dataset_name == 'CIFAR10DVS':
        net = models.CIFAR10DVSNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                    alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                    number_layer=number_layer).to(device)
    elif dataset_name == 'ASLDVS':
        net = models.ASLDVSNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                number_layer=number_layer).to(device)
    elif dataset_name == 'DVS128Gesture':
        net = models.DVS128GestureNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                        alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                        number_layer=number_layer).to(device)
            


    print(net1)
    # 使用Adam优化器
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=T_max)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=T_max)

    if check_point is not None:
        optimizer1.load_state_dict(check_point['optimizer'])
        scheduler1.load_state_dict(check_point['scheduler'])
        log_data_list = check_point['log_data_list']
        del check_point
    else:
        log_data_list = []

    if log_data_list.__len__() > 0 and not os.path.exists(log_dir):
        # 当tensorboard出现了飞线，可以人为删除log_dir，在这种情况下
        # log_data_list非空，而log_dir不存在
        # 可以从log_data_list将tensorboard重写，避免飞线
        rewrite_tb = True
    else:
        rewrite_tb = False
    writer = SummaryWriter(log_dir)
    if rewrite_tb:
        for i in range(log_data_list.__len__()):
            for item in log_data_list[i]:
                writer.add_scalar(item[0], item[1], item[2])

    if net1.epoch != 0:
        net1.epoch += 1
    ckpt_time = time.time()
    
    #清空原来的max_test值
    net1.max_test_accuracy=0
    if args.KD == False:
        if args.portion == 'all':
            runs_dir = 'runs/SNN_PLIF_{}_direct_all_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        elif args.portion == 'fc':
            runs_dir = 'runs/SNN_PLIF_{}_direct_fc_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        elif args.portion == 'conv':
            runs_dir = 'runs/SNN_PLIF_{}_direct_conv_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        if args.portion == 'all':
            runs_dir = 'runs/SNN_PLIF_{}_KD_all_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        elif args.portion == 'fc':
            runs_dir = 'runs/SNN_PLIF_{}_KD_fc_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        elif args.portion == 'conv':
            runs_dir = 'runs/SNN_PLIF_{}_KD_conv_{}'.format(dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    my_writer = SummaryWriter(log_dir=os.path.join(log_dir_prefix, runs_dir))

    for net1.epoch in tqdm(range(net1.epoch, max_epoch),desc="Processing"):

        start_time = time.time()

        log_data_list.append([])
        print(
            f'log_dir={log_dir}, max_test_accuracy={net1.max_test_accuracy}, train_times={net1.train_times}, epoch={net1.epoch}')
        print(args)
        print(argv)

        net1.train()
        net2.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        total_crdloss = 0
        for img, label in train_data_loader:
            param_copy(net1, net2)
            if quantize == True:
                net_quantize(net2)
            img = img.to(device)
            label = label.to(device)

            mid_outputs, preds = net1.forward_hook(img)
            mid_outputs_q, preds_q = net2.forward_hook(img)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            out_spikes_counter = net2(img)
            out_spikes_counter_frequency = out_spikes_counter / net2.T

            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
            crd_loss = CRD_loss(mid_outputs, mid_outputs_q)
            total_loss += loss.item()
            total_crdloss += crd_loss.item()


            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=1.0)
            
            grad_copy(net2, net1)

            if KD == True:
                crd_loss.backward()
                # 裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=1.0)

            optimizer1.step()
            functional.reset_net(net1)
            functional.reset_net(net2)

            accuracy = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
            total_accuracy += accuracy * img.size(0)  # 乘以样本数
            total_samples += img.size(0)  # 累计样本数
            if net1.train_times % 256 == 0:
                log_data_list[-1].append(('train_accuracy', accuracy, net1.train_times))
                log_data_list[-1].append(('train_loss', loss.item(), net1.train_times))
            net1.train_times += 1

        # 计算平均损失和平均准确率
        avg_loss = total_loss / total_samples  # 总损失 / 样本数
        avg_crdloss = total_crdloss / total_samples
        avg_accuracy = total_accuracy / total_samples  # 总准确率 / 样本数
        print('train_loss', avg_loss, 'crd_loss', avg_crdloss, 'train_accuracy', avg_accuracy)
        scheduler1.step()

        net1.eval()
        net2.eval()
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                param_copy(net1, net2)
                if quantize == True:
                    net_quantize(net2)

                img = img.to(device)
                label = label.to(device)
                out_spikes_counter = net2(img)
                correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net1)
                functional.reset_net(net2)
            test_accuracy = correct_sum / test_sum
            print('test_accuracy', test_accuracy)
            log_data_list[-1].append(('test_accuracy', test_accuracy, net1.epoch))
            if use_plif:
                plif_idx = 0
                for m in net1.modules():
                    if isinstance(m, models.PLIFNode):
                        log_data_list[-1].append(('w' + str(plif_idx), m.w.item(), net1.train_times))
                        plif_idx += 1

            print('Writing....')
            for item in log_data_list[-1]:
                writer.add_scalar(item[0], item[1], item[2])
            if net1.max_test_accuracy <= test_accuracy:
                print('save model with test_accuracy = ', test_accuracy)
                net1.max_test_accuracy = test_accuracy
                torch.save({
                    'net': net1,
                    'optimizer': optimizer1.state_dict(),
                    'scheduler': scheduler1.state_dict(),
                    'log_data_list': log_data_list
                }, check_point_max_path)
            with open(running_data_file, 'a') as f:
                f.write(f"epoch: {net1.epoch}, test_accuracy: {test_accuracy}, train_loss: {avg_loss}, train_accuracy: {avg_accuracy}, crd_loss: {avg_crdloss}")
                f.write('\n')
                f.flush()
            # 保存最新模型
            torch.save({
                'net': net1,
                'optimizer': optimizer1.state_dict(),
                'scheduler': scheduler1.state_dict(),
                'log_data_list': log_data_list
            }, check_point_path)

            print('Written.')
            my_writer.add_scalar('train_acc', accuracy, net1.epoch)
            my_writer.add_scalar('test_acc', test_accuracy, net1.epoch)
            my_writer.add_scalar('train_loss', total_loss, net1.epoch)
            my_writer.add_scalar('crd_loss', total_crdloss, net1.epoch)

            my_writer.flush()

            speed_per_epoch = time.time() - start_time
            print('speed per epoch', speed_per_epoch)


        
    writer.close()
