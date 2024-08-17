import os
import sys
# 获取当前文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 将当前文件路径加入到 sys.path
sys.path.append(current_path)
sys.path.append(current_path + "/..")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import *
from utils import get_model, get_parser
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import default_loader
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, random_split
import torchattacks
from tqdm import tqdm
from torchvision.utils import save_image

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
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    return testloader

def process_images(model, device):
    # atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
    atk = torchattacks.MIFGSM(model, eps=8/255, steps=10, decay=1.0)

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
        adv_img = atk(data, label)
        output_img_path = path[0].replace("clean_cls_samples", "adv_samples")
        
        output_dir_path = os.path.dirname(output_img_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        save_image(adv_img, output_img_path)

if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = get_model(args, device)
    net = net.to(device)

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.arch}.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    net = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), net)

    process_images(net, device)