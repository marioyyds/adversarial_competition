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
    parser = argparse.ArgumentParser(description='Eval model')
    parser.add_argument('--arch', default="resnet18", choices=['resnet18', 'resnet50', 'vit'],help='the network architecture')
    parser.add_argument('--gpu', default="0, 1", type=str, help='which gpus are available')
    parser.add_argument('--test_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/adv_samples', type=str, help='test set path')
    args = parser.parse_args()
    return args

def get_loader(args):
    # Data
    print('==> Preparing data..')

    # 设置图像的路径
    test_data_path = args.test_set

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 使用ImageFolder加载数据
    # train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform_train)
    # test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform_test)

    # 使用自定义的Dataset，这会将数据集提前加载到内存以提高速度
    test_dataset = CustomDataset(root=test_data_path, transform=transform_test)
   
    print(f"测试集大小: {len(test_dataset)}")

    # 创建DataLoader用于加载数据
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

    return testloader

def get_model(args):

    print('==> Building model..')
    if args.arch == "resnet18":
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 20)
    elif args.arch == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, 20)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model
  

if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    testloader = get_loader(args)
    
    net = get_model(args)
    net = net.to(device)

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.arch}.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
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