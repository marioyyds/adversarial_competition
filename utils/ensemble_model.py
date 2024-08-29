import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms


class Ensemble_logits(nn.Module):

    def __init__(self, model_list, input_size=[], prob=0.7, mode="nearest"):
        super(Ensemble_logits, self).__init__()
        self.model_list = model_list
        self.length = len(self.model_list)
        self.prob = prob
        self.mode = mode
        self.input_size = input_size # resnet18, resnet34, inceptionv3

    def resize(self, input, target_size):
        return F.interpolate(input, size=target_size, mode=self.mode)
    
    def random_crop_pad(self,input_tensor, input_size, target_scale=0.1):
        # random crop and pad
        target_size = int(input_size * (target_scale + 1.0))
        small_size = int(input_size * (1.0 -target_scale))
        rnd = np.floor(np.random.uniform(small_size, target_size, size=())).astype(np.int32).item()
        x_resize = self.resize(input_tensor, rnd)
        h_rem = target_size - rnd
        w_rem = target_size - rnd
        pad_top = np.floor(np.random.uniform(0, h_rem, size=())).astype(np.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = np.floor(np.random.uniform(0, w_rem, size=())).astype(np.int32).item()
        pad_right = w_rem - pad_left
        padded = F.pad(x_resize, (int(pad_top), int(pad_bottom),int(pad_left), int(pad_right), 0, 0, 0, 0))
        return padded
    
    def flip_image(self,input_tensor):
        """
        flip image（Horizontal or Vertical）。

        :param input_tensor: input image tensor
        :param horizontal: True，Horizontal；False，Vertical
        :return: image tensor after flipping
        """
        # horizontal or vertical
        if torch.rand(1) <= 0.5:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3)  
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.3)  
            ])
        
        return transform(input_tensor)
    
    def rotate_image(self,input_tensor, angle=18):
        """
        Rotate images at random angles
        
        :param input_tensor: input image tensor
        :param angle: rotate degree
        :return: image tensor after rotation
        """
        # 旋转图像
        transform = transforms.Compose([
            transforms.RandomRotation(angle)
        ])
        return transform(input_tensor)        


    
    def input_diversity(self, input_tensor, input_size):
        rotated_tensor = self.rotate_image(input_tensor)
        flipped_images = self.flip_image(rotated_tensor)
        padded_tensor = self.random_crop_pad(flipped_images, input_size, target_scale=0.1)
        
        if torch.rand(1) <= self.prob:
            return self.resize(padded_tensor, input_size)
        else:
            return self.resize(input_tensor, input_size)
        
       

    def forward(self, x, diversity=False):
        if diversity:
            output = torch.cat([self.model_list[idx](self.input_diversity(input_tensor=x, input_size=self.input_size[idx])).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)
        else:
            output = torch.cat([self.model_list[idx](self.resize(x, target_size=self.input_size[idx])).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)