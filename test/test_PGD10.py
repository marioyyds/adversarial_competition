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


# PGD10攻击的实现
def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, num_iter=10):
    images = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)
    loss = torch.nn.CrossEntropyLoss()

    for i in range(num_iter):
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        images = torch.clamp(images + eta, min=0, max=1).detach_()
        images.requires_grad = True

    return images

# 递归处理文件夹及其子文件夹
def process_images(input_dir, output_base_dir, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 修改为网络所需尺寸
        transforms.ToTensor(),
    ])
    
    for root, dirs, files in os.walk(input_dir):
        # 获取相对于 input_dir 的路径
        relative_path = os.path.relpath(root, input_dir)
        output_dir = os.path.join(output_base_dir, relative_path)

        # 创建对应的输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 遍历文件夹中的所有图片文件
        for img_file in tqdm(files):
            img_path = os.path.join(root, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"无法打开图片 {img_path}，错误: {e}")
                continue

            img_tensor = transform(img).unsqueeze(0).to(device)

            # 假设我们有标签，或者对目标模型使用相应的推理获得标签
            labels = torch.tensor([0]).to(device)  # 修改为实际标签
            
            # 使用PGD攻击
            adv_img = pgd_attack(model, img_tensor, labels)

            # 保存攻击后的图片
            output_img_path = os.path.join(output_dir, img_file)
            save_image(adv_img, output_img_path)

# 示例使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=> creating model ")
netClassifier = resnet18()
netClassifier.fc = nn.Linear(netClassifier.fc.in_features, 20)

checkpoint = torch.load('./checkpoint/resnet18.pth')
new_checkpoint = {}
for key, val in  checkpoint['net'].items():
    key = key.replace('module.', '')
    new_checkpoint[key] = val
netClassifier.load_state_dict(new_checkpoint)
netClassifier.to(device)

netClassifier = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), netClassifier)

input_dir = '/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples'  # 原始图片文件夹
output_dir = '/data/hdd3/duhao/data/datasets/attack_dataset/adv_samples'  # 对抗图片保存文件夹

process_images(input_dir, output_dir, netClassifier)
