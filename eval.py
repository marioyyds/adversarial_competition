import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from utils import progress_bar, get_model, get_parser
from utils import CustomDataset

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

if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    testloader = get_loader(args)
    
    net = get_model(args, device)
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