#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import torch
import numpy as np
import skimage.io as io

def save_one_img(tensor, save_path):
    # save single picture
    if len(tensor.shape) == 4:
        # batch, channel, height, weight
        img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(np.uint8(img * 255.0)).convert('RGB')
    img.save(save_path)

def load_one_img(img_path):
    img = torch.from_numpy(io.imread(img_path))
    img = img.permute(2, 0, 1) / 255.0
    return img
