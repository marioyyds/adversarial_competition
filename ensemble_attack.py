import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
from PIL import Image
import scipy.stats as st
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import default_loader
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.utils import save_image
from tqdm import tqdm

os.environ['TORCH_HOME']='~/.cache/torch/'

from utils import get_architecture, Ensemble_logits

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

def tensor2img(input_tensor, save_dir, save_name):

    if input_tensor.is_cuda == True:
        input_tensor = input_tensor.cpu()

    input_tensor = input_tensor.permute(0, 2, 3, 1).data.numpy()
    for i in range(input_tensor.shape[0]):
        Image.fromarray((input_tensor[i] * 255).astype(np.uint8)).save('{}/{}'.format(save_dir, save_name[i]))
        print('{} saved in {}.'.format(save_name[i], save_dir))

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

def _get_norm_batch(x, p):
    return x.abs().pow(p).sum(dim=[1, 2, 3], keepdims=True).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return x / norm

if __name__ == '__main__':
    clean_cls_samples_path = "/data2/huhongx/adversarial_competition/attack_dataset/clean_cls_samples"
    dataset = CustomDataset(clean_cls_samples_path, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Model to generate adv examples")
    # delete one model if OOM (all 8 models need around 12G),now need 5.5G    
    # model_name_list = ['resnet50.a1_in1k', 'inception_v3', 'efficientnet_b0.ra4_e3600_r224_in1k', 'resnet18', 'resnet18-at']
    model_name_list = ['resnet50.a1_in1k', 'resnet50.a1_in1k-at', 'inception_v3', 'efficientnet_b0.ra4_e3600_r224_in1k',  'resnet18', 'resnet18-at']
    model_input_size = [224, 224, 229, 224, 224, 224]
    print(model_name_list)
    model_list = []
    for model_name in model_name_list:
        temp_model = get_architecture(model_name, device).cuda()
        if os.path.exists(f'./checkpoint/{model_name}.pth'):
            temp_checkpoint = torch.load(f'./checkpoint/{model_name}.pth')
        elif os.path.exists(f'./checkpoint/at_train/{model_name}.pth'):
            temp_checkpoint = torch.load(f'./checkpoint/at_train/{model_name}.pth')
        else:
            raise ValueError("权重不存在！")
        
        temp_model.load_state_dict(temp_checkpoint['net'])
        temp_model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), temp_model)
        temp_model.eval()
        model_list.append(temp_model)

    # The Ensemble_logits Module use input diversity with 0.7 probability and nearest mode (interpolate) for each model
    ensemble_model = Ensemble_logits(
        model_list=model_list, input_size=model_input_size, prob=0.7, mode="nearest").cuda()
    ensemble_model.eval()

    # eval model
    print("Model to predict adv examples's confidence")
    model_name_list = ['mobilenet_v2', 'resnet101.a1h_in1k']
    model_input_size = [224, 224]
    print(model_name_list)
    model_list = []
    for model_name in model_name_list:
        temp_model = get_architecture(model_name, device).cuda()
        temp_checkpoint = torch.load(f'./checkpoint/{model_name}.pth')
        temp_model.load_state_dict(temp_checkpoint['net'])
        temp_model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), temp_model)
        temp_model.eval()
        model_list.append(temp_model)

    eval_model = Ensemble_logits(
        model_list=model_list, input_size=model_input_size, prob=0.7, mode="nearest").cuda()
    eval_model.eval()

    for inputs, (targets, path) in tqdm(data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.shape[0]

        g = torch.zeros_like(inputs).cuda() # for momentum update
        delta = torch.zeros_like(inputs).cuda() # init perturbation
        output_delta = torch.zeros_like(delta)
        mask = torch.ones((batch_size, )).bool()
        
        # eps_list = [4.0 / 255, 6.0 / 255, 8.0 / 255, 12.0 / 255, 16.0 / 255, 32.0 / 255, 64.0 / 255] 
        eps_list = [32.0 / 255, 48.0 / 255]
        # eps_list = [32.0 / 255]
        # eps_list = [64.0 / 255] 
        for eps in eps_list:
            if eps <= 8.0 / 255:
                num_steps = 10
            elif eps <= 16.0 / 255:
                num_steps = 20
            else:
                num_steps = 50
            step_size = (eps * 1.25) / num_steps
            delta = Variable(delta.data, requires_grad=True)
            for _ in range(num_steps):
                delta.requires_grad_()
                adv = inputs + delta
                adv = torch.clamp(adv, 0, 1)
                adv = F.conv2d(adv, weight=get_kernel(5),stride=(1, 1), groups=3, padding=(5 - 1) // 2) # use gaussian filter on adv samples
                with torch.enable_grad():
                    ensem_logits = ensemble_model(adv, diversity=True)
                    loss = F.cross_entropy(ensem_logits, targets, reduction="none")
                PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
                # gaussian filter with 5x5 kernel on gradient 
                PGD_grad = F.conv2d(PGD_grad, weight=get_kernel(5),stride=(1, 1), groups=3, padding=(5 - 1) // 2)
                PGD_noise = normalize_by_pnorm(PGD_grad, p=1)
                g[mask] = g[mask] * 0.8 + PGD_noise[mask]
                delta = Variable(delta.data + step_size * torch.sign(g), requires_grad=True)
                delta = Variable(torch.clamp(delta.data, -eps, eps), requires_grad=True)

            # reset the memoried grad to zero when restart with a bigger eps
            g *= 0.0

            with torch.no_grad():
                tmp = inputs + delta
                tmp = torch.clamp(tmp, 0, 1)
                output = eval_model(tmp, diversity=False).data

            prob = F.softmax(output, dim = 1)
            conf = prob[np.arange(batch_size), targets.long()]
            # if the transfer confidence is still bigger than 1%, it may need bigger eps
            mask = (conf >= 0.01)

            indices = torch.where(conf <= 0.01)[0]
            output_delta[indices] = torch.clone(delta[indices])

            if mask.sum() == 0:
                break
        zero_indices = [i for i in range(output_delta.size(0)) if torch.all(output_delta[i] == 0)]
        output_delta[zero_indices] = torch.clone(delta[zero_indices])
        # print("Attack max eps level: {} finished, conf: {}".format(eps, conf))
        X_pgd = Variable(inputs + output_delta, requires_grad=False)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
        with torch.no_grad():
            ensem_logits = ensemble_model(X_pgd, diversity=False)
        out_X_pgd = ensem_logits.detach()
        err_pgd = (out_X_pgd.max(1)[1] != targets.data).float().sum()
        # print("batch size: {}, attacked: {}".format(batch_size, err_pgd.item()))

        output_data = X_pgd.clone()

        for img, save_path in zip(output_data, path):
            save_path = save_path.replace("clean_cls_samples", "adv_samples")
            output_dir_path = os.path.dirname(save_path)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            save_image(img, save_path)
    
