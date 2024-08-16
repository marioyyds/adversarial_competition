import os
import sys

from torchvision.datasets.folder import default_loader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from models import *
from utils import progress_bar
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.utils.data as data
from torchvision.models import *

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root: str, transform: Union[Callable[..., Any], None] = None,
                 target_transform: Union[Callable[..., Any], None] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Union[Callable[[str], bool], None] = None, 
                 num_threads: int = 32):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.data = []
        self.num_threads = num_threads
        self._load_data()
    
    def _load_data(self):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self.loader, sample_path): target for sample_path, target in self.samples}
            for future in as_completed(futures):
                img = future.result()
                target = futures[future]
                self.data.append((img, target))
    
    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:

        return self.transform(self.data[index][0]), self.data[index][1]

def get_parser():
    parser = argparse.ArgumentParser(description='Classfication Model Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arch', default="resnet18", choices=['resnet18', 'resnet34', 'resnet50', 'vit', 'inception_v3'],help='the network architecture')
    parser.add_argument('--gpu', default="0", type=str, help='which gpus are available')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--train_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/phase1/train_set', type=str, help='train set path')
    parser.add_argument('--test_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/phase1/test_set', type=str, help='test set path')
    args = parser.parse_args()
    return args

def get_loader(args):
    # Data
    print('==> Preparing data..')

    # 设置图像的路径
    train_data_path = args.train_set
    test_data_path = args.test_set

    if args.arch in ["resnet18", "vit", "resnet34", "resnet50"]:
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
    elif args.arch == "inception_v3":
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
        raise ValueError(f"Unsupported architecture: {args.arch}")
        # 打印一些信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建DataLoader用于加载数据
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader

def get_model(args):

    print('==> Building model..')
    if args.arch == "resnet18":
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif args.arch == "resnet34":
        model = resnet34()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif args.arch == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif args.arch == "vit":
        model = vit_b_16()
        model.heads = nn.Linear(model.heads.head.in_features, 20) 
    elif args.arch == "inception_v3":
        model = inception_v3()
        model.fc = nn.Linear(model.fc.in_features, 20)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model


# Training
def train(epoch, args, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.arch == "inception_v3":
            # InceptionV3 有两个输出，需要分别计算损失
            outputs, aux_outputs = net(inputs)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(aux_outputs, targets)
            loss = loss1 + 0.4 * loss2  # 加权损失
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, args, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.arch}.pth')
        best_acc = acc


if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader = get_loader(args)
    
    net = get_model(args)
    net = net.to(device)
 
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/{args.arch}.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, args, device)
        test(epoch, args, device)
        scheduler.step()