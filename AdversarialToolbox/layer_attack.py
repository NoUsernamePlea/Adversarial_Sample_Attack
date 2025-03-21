#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from torch import nn
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import save_one_img
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pickle
from torchattacks import attacks
from torchattacks.tools.seed import random_seed
import time

def main():
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    for mname in model_names:
        if mname == 'resnet50':
            layers = mzoo.res50_feature_layers
        elif mname == 'densenet121':
            layers = mzoo.dense121_feature_layers
        elif mname == 'vgg19':
            layers = mzoo.vgg19_feature_layers
        else:
            continue
    # for mname in ['resnet50', 'densenet121', 'wide_resnet50_2']:
        for split_index, split_layer in enumerate(layers): 
            print('model {} with {} generates adversarial examples...'.format(mname, split_layer))
    
            adv_output_dir = os.path.join(layer_output_path, str(split_layer), mname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)
    
            for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):
    
                adv_output_dir = os.path.join(layer_output_path, str(split_layer), mname, dname)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('1. dataset {} is attacked...'.format(dname)) 
                ds = dzoo.load_dataset(dname, dpath)
                label_space = list(ds.class_to_idx.values())
    
                model = mzoo.pick_model(mname)
                feature_model, decision_model = mzoo.default_split(mname, split_index=split_index)
                model = model.cuda()
                model.eval()
                if feature_model is not None:
                    feature_model = feature_model.cuda()
                    decision_model = decision_model.cuda()
                    feature_model.eval()
                    decision_model.eval()
    
                for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                    random_seed()
    
                    # if attack_name not in ['LBAP']:
                    if attack_name not in ['LBAP-Conv']:
                        continue
    
                    adv_output_dir = os.path.join(layer_output_path, str(split_layer), mname, dname, attack_name)
                    if not os.path.exists(adv_output_dir):
                        os.makedirs(adv_output_dir)
        
                    print('2.{} attack method {} is attacking...'.format(i, attack_name))
                    if attack_name == 'LBAP':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.lbap.LBAP(model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n=10)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'LBAP-Conv':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.llbap.LLBAP(model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n=10, emb_aug_type='conv')
                        # targeted
                        attack.set_mode_targeted_by_label()
                    else:
                        # raise 'Invalid attack method!!!'
                        continue
    
                    # begin to attack
                    adv_confidences = {} 
                    start = time.time()
                    aux_samples = []
                    for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                        feature = feature.unsqueeze(0).cuda()
                        source = torch.LongTensor([label]).cuda()

                        if len(aux_samples) == 0:
                            for si in range(10):
                                fs, lab = ds[si]
                                aux_samples.append(fs.unsqueeze(0).cuda())
                        else:
                            aux_samples.pop(0)
                            aux_samples.append(feature.clone())
    
                        fname_basename = os.path.basename(fname)
                        (_, target) = attack_targets[fname_basename]
                        target = torch.LongTensor([target]).cuda()
                        adv_output_file = os.path.join(adv_output_dir, fname_basename)
    
                        adv_feature = attack(feature, target, source_labels=source,
                                feature_model=feature_model, pred_model=decision_model,
                                aux_samples=aux_samples) 
                        save_one_img(adv_feature.detach().cpu(), adv_output_file)
    
                        adv_confidence = F.softmax(model(adv_feature), dim=1)
                        adv_confidences[fname_basename] = adv_confidence.detach().cpu().numpy()
                    end = time.time()
    
                    adv_output_time = os.path.join(adv_output_dir, 'time.npy')
                    with open(adv_output_time, 'w') as f:
                        f.write(str(end-start) + '\n')

                    adv_output_confidence = os.path.join(adv_output_dir, 'confidence.npy')
                    with open(adv_output_confidence, 'wb') as fp:
                        pickle.dump(adv_confidences, fp)

if __name__ == '__main__':
    main()
