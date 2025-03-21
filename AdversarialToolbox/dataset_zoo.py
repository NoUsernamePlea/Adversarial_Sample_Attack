#!/usr/bin/env python
# coding=utf-8
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class DatasetZoo:
    def __init__(self):
        self.imagenet_composer = Compose([Resize([299, 299]), 
                                          ToTensor()])

    def load_dataset(self, dataset_name, dataset_path):
        if dataset_name == 'imagenet':
            idx2label = ['{:03d}'.format(i) for i in range(1000)]
            old_dataset = datasets.ImageFolder(dataset_path, self.imagenet_composer)
            old_classes = old_dataset.classes

            label2idx = {}
            for i, item in enumerate(idx2label):
                label2idx[item] = i
                        
            new_dataset = datasets.ImageFolder(dataset_path, self.imagenet_composer, 
                                                   target_transform=lambda x: idx2label.index(old_classes[x]))
            new_dataset.classes = idx2label
            new_dataset.target_transform = lambda x: idx2label.index(old_classes[x])
            new_dataset.class_to_idx = label2idx
            ds = new_dataset
        else:
            raise 'Invalid dataset name {}!!!'.format(dataset_name)
        return ds
