import argparse
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import get_architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

class Dataset():
    def __init__(self, adv_path, clean_path, transform):
        self.adv_samples = datasets.ImageFolder(adv_path, transform=transform)
        self.clean_samples = datasets.ImageFolder(clean_path, transform=transform)
    
    def __len__(self) -> int:
        return len(self.clean_samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        return self.adv_samples.__getitem__(index), self.clean_samples.__getitem__(index)
    
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
    
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_kernel(kernel_size = 7):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3).transpose(2, 3, 0, 1)
    stack_kernel = torch.from_numpy(stack_kernel).cuda()
    return stack_kernel

parser = argparse.ArgumentParser(description='Evaluate locally')
parser.add_argument('--adv_sample', default='/data/hdd3/duhao/data/datasets/attack_dataset/adv_samples-64_8', type=str, help='adv samples path')
parser.add_argument('--clean_sample', default='/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples', type=str, help='clean samples path')
parser.add_argument('--balckbox_models', default=["resnet101.a1h_in1k", "resnet34"], type=str, help='black models')
args = parser.parse_args()

adv_sample = args.adv_sample
clean_sample = args.clean_sample

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
    ])

dataset = Dataset(adv_path=adv_sample, clean_path=clean_sample, transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=16)

balckbox_models = args.balckbox_models
print(f"blackbox models:{balckbox_models}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = []
for m in balckbox_models:
    temp_model = get_architecture(m, device).to(device)
    temp_checkpoint = torch.load(f'./checkpoint/{m}.pth')
    temp_model.load_state_dict(temp_checkpoint['net'])
    temp_model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), temp_model)
    temp_model.eval()
    models.append(temp_model)

ACTC_score = 0
ALD_score = 0
RGB_score = 0
miss_cla_num = [[0, 0] for m in balckbox_models]
with torch.no_grad():
    for ((adv_sample, _), (clean_sample, label)) in tqdm(dataloader):
        adv_sample = adv_sample.to(device)
        clean_sample = clean_sample.to(device)
        label = label.to(device)
        for inx, model in enumerate(models):
            # cal ACTC
            predictd = model(adv_sample)
            pro = nn.functional.softmax(predictd.squeeze(0), dim=0)[label]
            ACTC_score += pro.cpu().item()

            # cal ALD
            _, predictd_label = nn.functional.softmax(predictd.squeeze(0), dim=0).max(0)
            if predictd_label != label:
                distortion = (adv_sample - clean_sample).max() * 255
            else:
                distortion = 64
            ALD_score += distortion

            # cal RGB
            if predictd_label != label:
                miss_cla_num[inx][0] = miss_cla_num[inx][0] + 1
                gauss_blur_adv_sample = F.conv2d(adv_sample, weight=get_kernel(5),stride=(1, 1), groups=3, padding=(5 - 1) // 2)
                predictd = model(gauss_blur_adv_sample)
                _, predictd_label = nn.functional.softmax(predictd.squeeze(0), dim=0).max(0)
                if predictd_label != label:
                    miss_cla_num[inx][1] = miss_cla_num[inx][1] + 1



ACTC_score = ACTC_score / (len(dataloader) * len(models))
ALD_score = ALD_score / (len(dataloader) * len(models) * 64)
for mis in miss_cla_num:
    RGB_score += mis[1] / mis[0]
RGB_score = RGB_score / 5

final_score = 100 * ((1 - ACTC_score) + (1 - ALD_score) + RGB_score) / 3

print(f"(1-ACTC) score: {(1 - ACTC_score) * 100}")
print(f"(1-ALD) score: {(1 - ALD_score) * 100}")
print(f"RGB score: {RGB_score * 100}")
print(f"final score: {final_score}")