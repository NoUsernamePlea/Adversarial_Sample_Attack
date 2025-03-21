#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import load_one_img
from attack_metric import AttackMetric
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

with open(attack_book, 'r') as fp:
    attack_targets = json.load(fp)

def evaluate(evaluation_file, src_model_name, target_model_name, attakc_method, dataset_name, dataset_path, arg):
    mzoo = ModelZoo() 

    # load target model
    model = mzoo.pick_model(target_model_name)
    model = model.cuda()
    model.eval()

    # prepare source adversarial examples
    adv_output_dir = os.path.join(gamma_output_path, str(arg), src_model_name, dataset_name, attack_method)
    white_confidence_file = [fname for fname in os.listdir(adv_output_dir) if fname.endswith('.npy')][0]
    with open(os.path.join(adv_output_dir, white_confidence_file), 'rb') as fp:
        white_confidence_dict = np.load(fp, allow_pickle=True)

    # begin to evaluate
    adv_imgs = [fname for fname in os.listdir(adv_output_dir) if not fname.endswith('.npy')]
    src_labels, target_labels, white_confidences, black_confidences = [], [], [], []
    for fname in tqdm(adv_imgs):
        fpath = os.path.join(adv_output_dir, fname)
        feature = load_one_img(fpath)
        feature = feature.unsqueeze(0).cuda()

        try:
            (src, target) = attack_targets[fname]
        except Exception as e:
            continue

        src_labels.append(src)
        target_labels.append(target)

        adv_logits = model(feature)
        adv_confidence = F.softmax(adv_logits, dim=1)
        white_confidences.append(white_confidence_dict[fname])
        black_confidences.append(adv_confidence.detach().cpu().numpy())
    black_confidences = np.concatenate(black_confidences)
    white_confidences = np.concatenate(white_confidences)
    src_labels = np.array(src_labels)
    target_labels = np.array(target_labels)

    am = AttackMetric(src_labels, target_labels, white_confidences, black_confidences)
    prefix = '{},{},{},{},{:.4f},{:.4f},'.format(arg, attack_method, src_model_name,
                                    target_model_name, am.error_rate(),
                                    am.targeted_success_rate())
    suffix = ''
    for n in [1000, 3000, 5000]: 
    # for n in [100, 500, 1000]: 
        is_n, utr = am.topn_untargeted_transfer_rate(n)
        if is_n:
            suffix = suffix + '{:.4f},'.format(utr)
        else:
            suffix = suffix + '{:.4f}*,'.format(utr)


    for n in [1000, 3000, 5000]: 
    # for n in [100, 500, 1000]: 
        is_n, ttr = am.topn_targeted_transfer_rate(n)
        if is_n:
            suffix = suffix + '{:.4f}'.format(ttr)
        else:
            suffix = suffix + '{:.4f}*'.format(ttr)

        if n != 5000:
        # if n != 1000:
            suffix = suffix + ','

    line = prefix + suffix
    # line = prefix
    print(line)
    with open(evaluation_file, 'a') as fp:
        fp.write(line + '\n')

if __name__ == '__main__':
    for src_mname in model_names:
    # for src_mname in ['resnet50']:
    # for src_mname in ['inception_v3']:
        for target_mname in victim_model_names:
            if src_mname == target_mname:
                continue
            for ds_name, ds_path in victim_datasets:
                for attack_method in baseline_attack_methods.keys():
                # for attack_method in ['BSI-FGSM']:
                # for attack_method in ['VNI-FGSM']:
                # for attack_method in ['VMI-FGSM']:
                # for attack_method in ['BAP']:
                # for attack_method in ['DI-FGSM']:
                # for attack_method in ['DDVNI-FGSM']:
                    # for lr in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
                    # for lr in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
                    # for arg in [2, 3, 4, 5, 6]:
                    for arg in [0.0, 0.05, 0.1, 0.15, 0.20, 0.25]:
                        if attack_method not in ['BAP']:
                            continue
                        evaluate(gamma_evaluation_file, src_mname, target_mname, attack_method, ds_name, ds_path, arg)
    print('===evaluate end===')

