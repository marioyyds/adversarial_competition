import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import *
from torchvision import datasets, transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import default_loader
from utils import *

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root: str, transform: Union[Callable[..., Any], None] = None,
                 target_transform: Union[Callable[..., Any], None] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Union[Callable[[str], bool], None] = None, 
                 num_threads: int = 32):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.data = self.imgs

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:

        return self.transform(default_loader(self.data[index][0])), self.data[index]

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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_false', help='enables cuda')
parser.add_argument('--target', type=int, default=0, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')
parser.add_argument('--max_count', type=int, default=50, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.1, help='patch size. E.g. 0.05 ~= 5% of image ')
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
# parser.add_argument('--train_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/phase1/train_set', type=str, help='train set path')
parser.add_argument('--train_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples', type=str, help='train set path')
parser.add_argument('--test_set', default='/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples', type=str, help='test set path')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_set = opt.train_set
test_set = opt.test_set

print("=> creating model ")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name_list = ['resnet50.a1_in1k', 'inception_v3', 'efficientnet_b0.ra4_e3600_r224_in1k', 'resnet18']
model_input_size = [224, 229, 224, 224]
print(model_name_list)
model_list = []
for model_name in model_name_list:
    temp_model = get_architecture(model_name, device).cuda()
    temp_checkpoint = torch.load(f'./checkpoint/{model_name}.pth')
    temp_model.load_state_dict(temp_checkpoint['net'])
    temp_model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), temp_model)
    temp_model.eval()
    model_list.append(temp_model)

# The Ensemble_logits Module use input diversity with 0.7 probability and nearest mode (interpolate) for each model
netClassifier = Ensemble_logits(
    model_list=model_list, input_size=model_input_size, prob=0.7, mode="nearest").cuda()
netClassifier.eval()

if opt.cuda:
    netClassifier.cuda()

print('==> Preparing data..')

train_loader = torch.utils.data.DataLoader(
    CustomDataset(train_set, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=False,
    num_workers=opt.workers)
 
test_loader = torch.utils.data.DataLoader(
    CustomDataset(test_set, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=False,
    num_workers=opt.workers)

def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, (path, labels)) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)
 
        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
     
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
 
        adv_x, mask, patch = attack(data, patch, mask, labels)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label != ori_label:
            success += 1

        if "clean_cls_samples" in path[0]:
            save_path = path[0].replace("clean_cls_samples", "train_adv_samples")
            output_dir_path = os.path.dirname(save_path)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            # plot adversarial image
            vutils.save_image(adv_x.data, save_path, normalize=True)
 
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))

    return patch

def test(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, (path, labels)) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)
        total += 1 
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
 
        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, 0, 1)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label != ori_label:
            success += 1

        save_path = path[0].replace("clean_cls_samples", "adv_samples")
        output_dir_path = os.path.dirname(save_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # plot adversarial image
        vutils.save_image(adv_x.data, save_path, normalize=True)
       
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success/total))

def attack(x, patch, mask, labels):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))

    # adv_x = torch.mul(1-mask,x) + torch.mul(mask,patch)
    adv_x = x + torch.mul(mask,patch)
    
    count = 0 
    epsilon = torch.tensor(32./255, dtype=torch.float32).cuda()
    step_size = torch.tensor(epsilon / 50, dtype=torch.float32).cuda()
    for i in range(50):
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        logits = netClassifier(adv_x)
        
        Loss = F.cross_entropy(logits, labels, reduction="none")
        adv_x_grad = torch.autograd.grad(Loss.sum(), [adv_x])[0].detach() 
       
        patch = adv_x + (1.25 * step_size * torch.sign(adv_x_grad))
        patch = torch.clamp(patch - x, -epsilon, epsilon)
        adv_x = x + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, 0, 1)
    patch = adv_x - x
    return adv_x, mask, patch 


if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
