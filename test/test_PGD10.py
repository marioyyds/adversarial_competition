import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from torchvision.models import *
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import default_loader
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, random_split

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
            futures = {executor.submit(self.loader, sample_path): (target, sample_path) for sample_path, target in self.samples}
            for future in as_completed(futures):
                img = future.result()
                target = futures[future]
                self.data.append((img, target))
    
    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:

        return self.transform(self.data[index][0]), self.data[index][1]


class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def pgd_attack(model, images, labels, device, eps=32. / 255., alpha=2. / 255., iters=10, advFlag=None, forceEval=True, randomInit=True):
    loss = nn.CrossEntropyLoss()

    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        if advFlag is None:
            if forceEval:
                model.eval()
            outputs = model(images + delta)
        else:
            if forceEval:
                model.eval()
            outputs = model(images + delta, advFlag)

        model.zero_grad()
        cost = loss(outputs, labels)
        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()

# 递归处理文件夹及其子文件夹
def process_images(model, device):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    # 使用ImageFolder加载数据
    # train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform_train)
    # test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform_test)

    test_dataset = CustomDataset(root='/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples', transform=transform_test)
   
    print(f"测试集大小: {len(test_dataset)}")

    # 创建DataLoader用于加载数据
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    
    for data, (label, path) in tqdm(testloader):
        data = data.to(device)
        label = label.to(device)
        adv_img = pgd_attack(model, data, label, device=device)
        output_img_path = path[0].replace("clean_cls_samples", "adv_samples")
        
        output_dir_path = os.path.dirname(output_img_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        save_image(adv_img, output_img_path)

# 示例使用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=> creating model ")
netClassifier = resnet18()
netClassifier.fc = nn.Linear(netClassifier.fc.in_features, 20)

if device == 'cuda':
    netClassifier = torch.nn.DataParallel(netClassifier)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/resnet18.pth')

netClassifier.load_state_dict(checkpoint["net"])
netClassifier.to(device)

netClassifier = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), netClassifier)

process_images(netClassifier, device)
