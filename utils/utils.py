'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch

import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.interpolation import rotate
import os
import sys

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from torchvision.models import *
from .custom_dataset import CustomDataset
from timm import create_model
import shutil
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
   
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask, patch.shape


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1,3,noise_dim,noise_dim)
    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
    
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask


def get_parser():
    parser = argparse.ArgumentParser(description='Classfication Model Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arch', default="resnet18", type=str,help='the network architecture')
    parser.add_argument('--gpu', default="1", type=str, help='which gpus are available')
    parser.add_argument('--epoch', default=200, type=int, help='how many epoch to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_worker', default=2, type=int, help='number of workers')
    parser.add_argument('--train_set', default='/data2/huhongx/adversarial_competition/attack_dataset/phase1/train_set', type=str, help='train set path')
    parser.add_argument('--test_set', default='/data2/huhongx/adversarial_competition/attack_dataset/phase1/test_set', type=str, help='test set path')
    parser.add_argument('--origin_sample_path', default='/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples', type=str, help='origin sample path')
    parser.add_argument('--attack_sample_path', default='/data/hdd3/duhao/code/adversarial_competition/adv_samples', type=str, help='attack sample path')
    parser.add_argument('--sample_type', default=1, type=int, help='attack sample path')

    args = parser.parse_args()
    return args

def get_loader(args):
    # Data
    print('==> Preparing data..')

    # 设置图像的路径
    train_data_path = args.train_set
    test_data_path = args.test_set

    if args.arch == "inception_v3":
        transform_train = transforms.Compose([
            transforms.Resize((299, 299)), 
            transforms.RandomCrop(299, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 使用ImageFolder加载数据
        # train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform_train)
        # test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform_test)

        # 使用自定义的Dataset，这会将数据集提前加载到内存以提高速度
        train_dataset = CustomDataset(root=train_data_path, transform=transform_train)
        test_dataset = CustomDataset(root=test_data_path, transform=transform_test)
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 使用ImageFolder加载数据
        # train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform_train)
        # test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform_test)

        # 使用自定义的Dataset，这会将数据集提前加载到内存以提高速度
        train_dataset = CustomDataset(root=train_data_path, transform=transform_train)
        test_dataset = CustomDataset(root=test_data_path, transform=transform_test)

        # 打印一些信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建DataLoader用于加载数据
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    return trainloader, testloader

def get_architecture(arch, device):
    arch = arch.split("-")[0]
    if arch == "resnet18":
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif arch == "resnet34":
        model = resnet34()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif arch == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif arch == "vit":
        model = vit_b_16()
        model.heads = nn.Linear(model.heads.head.in_features, 20) 
    elif arch == "inception_v3":
        model = inception_v3()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif arch == "vgg16":
        model = vgg16()
        model.classifier[6] = nn.Linear(in_features=4096, out_features=20)
    elif arch == "mobilenet_v2":
        model = mobilenet_v2()
        # 修改最后的全连接层，20类分类任务
        model.classifier[1] = nn.Linear(model.last_channel, 20)
    elif arch == "mobilenet_v3":
        model = mobilenet_v3_small(num_classes=20)
    elif arch == "swint":
        model = swin_t(weights='IMAGENET1K_V1')
        # 修改分类头，20个类别
        model.head = nn.Linear(model.head.in_features, 20)
    elif arch == "googlenet":
        # 加载预训练的 GoogLeNet 模型
        model = googlenet()
        model.fc = nn.Linear(model.fc.in_features, 20)
    else:
        model = create_model(arch, pretrained=False, num_classes=20)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model

def get_model(args, device):

    print('==> Building model..')
    model = get_architecture(args.arch, device)
    
    return model