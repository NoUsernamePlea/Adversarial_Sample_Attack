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
        print('model {} generates adversarial examples...'.format(mname))

        adv_output_dir = os.path.join(lint_output_path, mname)
        if not os.path.exists(adv_output_dir):
            os.makedirs(adv_output_dir)

        for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):

            adv_output_dir = os.path.join(lint_output_path, mname, dname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)

            print('1. dataset {} is attacked...'.format(dname)) 
            ds = dzoo.load_dataset(dname, dpath)
            label_space = list(ds.class_to_idx.values())

            model = mzoo.pick_model(mname)
            feature_model, decision_model = mzoo.default_split(mname)
            model = model.cuda()
            model.eval()
            if feature_model is not None:
                feature_model = feature_model.cuda()
                decision_model = decision_model.cuda()
                feature_model.eval()
                decision_model.eval()

            for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                random_seed()

                # if attack_name not in ['LINTSIT']:
                # if attack_name not in ['LINTDI-FGSM']:
                # if attack_name not in ['LINTTI-FGSM']:
                # if attack_name not in ['LINTDEM']:
                # if attack_name not in ['LINTAdmix']:
                if attack_name not in ['LINTSIT-Conv', 'LINTDEM-Conv', 'LINTAdmix-Conv']:
                    continue

                adv_output_dir = os.path.join(lint_output_path, mname, dname, attack_name)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('2.{} attack method {} is attacking...'.format(i, attack_name))
                if attack_name == 'LINTSIT':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_sit.LINTSIT(model,
                        eps=attack_args['eps'],
                        steps=attack_args['max_iter'],
                        alpha=alpha,
                        decay=attack_args['decay_factor'],
                        n_copies=10, mode='mixup')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTSIT-Conv':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_sit.LINTSIT(model,
                        eps=attack_args['eps'],
                        steps=attack_args['max_iter'],
                        alpha=alpha,
                        decay=attack_args['decay_factor'],
                        n_copies=10, mode='conv')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTDI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_difgsm.LINTDIFGSM(model,
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha,
                                diversity_prob=attack_args['diversity_prob'],
                                n_copies=10, mode='mixup')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTTI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_tifgsm.LINTTIFGSM(model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            len_kernel=attack_args['len_kernel'],
                            diversity_prob=attack_args['diversity_prob'],
                            n_copies=10, mode='mixup')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTDEM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_dem.LINTDEM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha, mode='mixup')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTAdmix':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_admix.LINTAdmix(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha,
                                ratio=attack_args['ratio'],
                                n=attack_args['n'], mode='mixup')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTDEM-Conv':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_dem.LINTDEM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha, mode='conv')
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'LINTAdmix-Conv':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.lint_admix.LINTAdmix(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha,
                                ratio=attack_args['ratio'],
                                n=attack_args['n'], mode='conv')
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
