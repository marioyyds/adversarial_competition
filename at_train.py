import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar, get_loader, get_parser, get_model
import torchattacks
from utils.custom_dataset import CustomDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
    train_data_path = args.train_set
    test_data_path = args.test_set

    if args.arch == "inception_v3":
        transform_train = transforms.Compose([
            transforms.Resize((299, 299)), 
            transforms.RandomCrop(299, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

# Training
def train(epoch, args, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    atk = torchattacks.PGD(nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), net), eps=8/255, alpha=2/255, steps=20)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = atk(inputs, targets)
        inputs = norm(inputs)       
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

    atk = torchattacks.PGD(nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), net), eps=8/255, alpha=2/255, steps=20)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = atk(inputs, targets)
        inputs = norm(inputs)
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
        torch.save(state, f'./checkpoint/at_train/{args.arch}.pth')
        best_acc = acc


if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader = get_loader(args)
    
    net = get_model(args, device)
    net = net.to(device)
 
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        if os.path.isfile(f'./checkpoint/{args.arch}.pth'):
            checkpoint = torch.load(f'./checkpoint/{args.arch}.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        elif os.path.isfile(f'./checkpoint/{args.arch.split("-")[0]}.pth'):
            checkpoint = torch.load(f'./checkpoint/{args.arch.split("-")[0]}.pth')
            net.load_state_dict(checkpoint['net'])
        else:
            raise ValueError("权重不存在！")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    print(start_epoch)
    for epoch in range(start_epoch, args.epoch):
        train(epoch, args, device)
        test(epoch, args, device)
        scheduler.step()