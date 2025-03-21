#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os
from torchdefenses.HGD.nips_deploy import res152_wide, inres, v3, resnext101
from torchvision import transforms
import torch.nn as nn
import torch
import pdb

class ModelHGD:
    def __init__(self, weight_path='../torchdefenses/pretrained/HGD'):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.weight_path = weight_path

    def pick_model(self, model_name):
        if model_name == 'resnet152':
            config, model = res152_wide.get_model()
            wpath = os.path.join(self.weight_path, 'denoise_res_015.ckpt')
        elif model_name == 'resnext101':
            config, model = resnext101.get_model()
            wpath = os.path.join(self.weight_path, 'denoise_rex_001.ckpt')
        elif model_name == 'inception_v3':
            config, model = v3.get_model()
            wpath = os.path.join(self.weight_path, 'denoise_incepv3_012.ckpt')
        elif model_name == 'inception_resnet_v2':
            config, model = inres.get_model()
            wpath = os.path.join(self.weight_path, 'denoise_inres_014.ckpt')
        else:
            raise 'Invalid model name!!!'

        checkpoint = torch.load(wpath)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                model.net
               )


if __name__ == '__main__':
    mz = ModelHGD()
    model_names = ['resnet152', 'resnext101', 'inception_v3', 'inception_resnet_v2'] 
    for mn in model_names:
        print(mn)
        model = mz.pick_model(mn)

