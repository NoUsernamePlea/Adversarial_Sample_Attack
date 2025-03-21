#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from model_zoo import ModelZoo
from model_hgd import ModelHGD
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import load_one_img
from attack_metric import AttackMetric
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchdefenses.Trans import defense

with open(attack_book, 'r') as fp:
    attack_targets = json.load(fp)

def evaluate(evaluation_file, holdout_model_name, ens_models, attakc_method, dataset_name, dataset_path, defense_name):
    mzoo = ModelZoo() 
    defense_method = defense.Defense(defense_name)

    # load target model
    model = mzoo.pick_model(holdout_model_name)
    model = model.cuda()
    model.eval()

    # prepare source adversarial examples
    adv_output_dir = os.path.join(hgd_output_path, holdout_model_name, dataset_name, attack_method)
    white_confidence_file = [fname for fname in os.listdir(adv_output_dir) if fname.endswith('.npy') and 'confidence' in fname][0]
    with open(os.path.join(adv_output_dir, white_confidence_file), 'rb') as fp:
        white_confidence_dict = np.load(fp, allow_pickle=True)

    # begin to evaluate
    adv_imgs = [fname for fname in os.listdir(adv_output_dir) if not fname.endswith('.npy')]
    src_labels, target_labels, white_confidences, black_confidences = [], [], [], []
    for fname in tqdm(adv_imgs):
        fpath = os.path.join(adv_output_dir, fname)
        feature = load_one_img(fpath)
        # defense
        feature = defense_method(feature)
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
    prefix = '{},{},{},{},{:.4f},{:.4f},'.format(defense_name, attack_method, 
                                    str(ens_models), holdout_model_name,
                                    am.error_rate(),
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
    model_names = ['resnet50', 'vgg19', 'densenet121', 'wide_resnet50_2']
    # for holdout_mname in ['ens_adv_inception_resnet_v2', 'adv_inception_v3']: 
    for holdout_mname in ['inception_v3']: 
        ens_models = np.setdiff1d(model_names, [holdout_mname])
        for ds_name, ds_path in victim_datasets:
            for defense_name in ['jpeg', 'quantize']:
                for attack_method in ['ENSDI-FGSM', 'ENSMI-FGSM', 'ENSBAP', 'ENSTI-FGSM', 'ENSSIT', 'ENSPoincare']:
                    evaluate(trans_evaluation_file, holdout_mname, ens_models, attack_method, ds_name, ds_path, defense_name)
    print('===evaluate end===')

