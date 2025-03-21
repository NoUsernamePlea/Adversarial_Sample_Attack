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

    start = True 
    for mname in model_names:
        print('model {} generates adversarial examples...'.format(mname))

        adv_output_dir = os.path.join(mlayer_output_path, mname)
        if not os.path.exists(adv_output_dir):
            os.makedirs(adv_output_dir)

        for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):

            adv_output_dir = os.path.join(mlayer_output_path, mname, dname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)

            print('1. dataset {} is attacked...'.format(dname)) 
            ds = dzoo.load_dataset(dname, dpath)
            label_space = list(ds.class_to_idx.values())

            model = mzoo.pick_model(mname)
            feature_model1, feature_model2, decision_model = mzoo.ushape_split(mname)
            model = model.cuda()
            model.eval()
            if feature_model1 is not None:
                feature_model1 = feature_model1.cuda()
                feature_model2 = feature_model2.cuda()
                decision_model = decision_model.cuda()
                feature_model1.eval()
                feature_model2.eval()
                decision_model.eval()

            for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                random_seed()

                # if attack_name not in ['LBAP-MMix', 'LBAP-MConv', 'LBAP-MMixConv', 'LBAP-MConvMix']:
                if attack_name not in ['LBAP-parallel', 'LBAP-mixconv', 'LBAP-convmix', 'LBAP-convconv', 
                        'LBAP-mixmix', 'LBAP-concatenate']:
                    continue

                # if mname == 'densenet121' and attack_name == 'LBAP-MConv':
                #    start = True

                if not start:
                    continue

                adv_output_dir = os.path.join(mlayer_output_path, mname, dname, attack_name)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('2.{} attack method {} is attacking...'.format(i, attack_name))
                '''
                if 'LBAP' in attack_name:
                    major, minor = attack_name.split('-')
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lbap.LBAP(model,
                        eps=attack_args['eps'],
                        steps=attack_args['max_iter'],
                        alpha=alpha,
                        decay=attack_args['decay_factor'],
                        n=10, mode=minor)
                    # targeted
                    attack.set_mode_targeted_by_label()
                '''
                if 'LBAP' in attack_name:
                    major, minor = attack_name.split('-')
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.llbap.LLBAP(model,
                        eps=attack_args['eps'],
                        steps=attack_args['max_iter'],
                        alpha=alpha,
                        decay=attack_args['decay_factor'],
                        n=10, emb_aug_type=minor)
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
                            feature_models=[feature_model1, feature_model2], pred_model=decision_model,
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
