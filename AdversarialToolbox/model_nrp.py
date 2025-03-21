#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os
from torchdefenses.NRP.networks import NRP, NRP_resG 
from torchvision import transforms
import torch.nn as nn
import torch
import pdb

class Perturb(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 16 / 255

    def forward(self, img):
        img_m = img + torch.randn_like(img) * 0.05
        img_m = torch.min(torch.max(img_m, img - self.eps), img + self.eps)
        img_m = torch.clamp(img_m, 0.0, 1.0)
        return img_m


class ModelNRP:
    def __init__(self, weight_path='../torchdefenses/pretrained/NRP', dynamic=False):
        self.weight_path = weight_path
        self.dynamic = dynamic

    def pick_model(self, model_name):
        if model_name == 'nrp':
            model = NRP(3, 3, 64, 23) 
            wpath = os.path.join(self.weight_path, 'NRP.pth')
        elif model_name == 'nrp_resg':
            model = NRP_resG(3, 3, 64, 23) 
            wpath = os.path.join(self.weight_path, 'NRP_resG.pth')
        else:
            raise 'Invalid model name!!!'

        checkpoint = torch.load(wpath)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        if self.dynamic:
            return nn.Sequential(
                       Perturb(),
                       model
                   )
        else:
            return model


if __name__ == '__main__':
    mz = ModelNRP()
    model_names = ['nrp', 'nrp_resg']
    for mn in model_names:
        print(mn)
        model = mz.pick_model(mn)

