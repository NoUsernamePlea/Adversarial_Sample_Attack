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

def main():
    device = (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
    for mname in victim_model_names: 
        print('model {} generates adversarial examples...'.format(mname))

        adv_output_dir = os.path.join(ens_output_path, mname)
        if not os.path.exists(adv_output_dir):
            os.makedirs(adv_output_dir)

        for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):

            adv_output_dir = os.path.join(ens_output_path, mname, dname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)

            print('1. dataset {} is attacked...'.format(dname)) 
            ds = dzoo.load_dataset(dname, dpath)
            label_space = list(ds.class_to_idx.values())

            holdout_model = mzoo.pick_model(mname)
            holdout_model.eval()
            holdout_model = holdout_model.to(device)

            ens_model_names = np.setdiff1d(model_names, [mname])
            ens_models = mzoo.pick_model_pool(ens_model_names)
            for md in ens_models:
                md.eval()
            ens_models = [md.to(device) for md in ens_models]

            '''
            feature_models, decision_models = [], []
            for model_name in ens_model_names:
                feature_model, decision_model = mzoo.default_split(model_name, split_index=-1)
                feature_model = feature_model.to(device)
                decision_model = decision_model.to(device)
                feature_model = feature_model.eval()
                decision_model = decision_model.eval()

                feature_models.append(feature_model)
                decision_models.append(decision_model)
            '''

            for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                random_seed()

                attack_args['max_iter'] = 50
                alpha = 2 / 255


                # skip some methods
                # if attack_name not in ['ENSMI-FGSM', 'ENSNI-FGSM', 'ENSDEM', 'ENSAdmix', 'ENSSIT', 'ENSDI-FGSM', 'ENSTI-FGSM', 'ENSLBAP']:
                # if attack_name not in ['ENSDEM', 'ENSAdmix', 'ENSLBAP']:
                if attack_name not in ['ENSSIT']:
                    continue

                adv_output_dir = os.path.join(ens_output_path, mname, dname, attack_name)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('2.{} attack method {} is attacking...'.format(i, attack_name))
                if attack_name == 'ENSDI-FGSM':
                    attack = attacks.ens_difgsm.ENSDIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            diversity_prob=attack_args['diversity_prob'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSMI-FGSM':
                    attack = attacks.ens_mifgsm.ENSMIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha)
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSLBAP':
                    alpha = 1.0 
                    attack = attacks.ens_lbap.ENSLBAP(holdout_model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n=40)
                    #        n=attack_args['n'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSTI-FGSM':
                    attack = attacks.ens_tifgsm.ENSTIFGSM(holdout_model, eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            len_kernel=attack_args['len_kernel'],
                            diversity_prob=attack_args['diversity_prob'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSSIT':
                    attack = attacks.ens_sit.ENSSIT(holdout_model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n_copies=40)
                    #        n=attack_args['n'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSDEM':
                    attack = attacks.ens_dem.ENSDEM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha)
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSAdmix':
                    attack = attacks.ens_admix.ENSAdmix(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            ratio=attack_args['ratio'],
                            n=attack_args['n'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                else:
                    # raise 'Invalid attack method!!!'
                    continue

                # begin to attack
                adv_confidences = {} 
                aux_samples = []
                for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                    feature = feature.unsqueeze(0).to(device)
                    source = torch.LongTensor([label]).to(device)

                    if len(aux_samples) == 0:
                        for si in range(10):
                            fs, lab = ds[si]
                            aux_samples.append(fs.unsqueeze(0).cuda())
                    else:
                        aux_samples.pop(0)
                        aux_samples.append(feature.clone())

                    fname_basename = os.path.basename(fname)
                    (_, target) = attack_targets[fname_basename]
                    target = torch.LongTensor([target]).to(device)
                    adv_output_file = os.path.join(adv_output_dir, fname_basename)

                    adv_feature = attack(feature, target, source_labels=source,
                                # feature_models=feature_models, pred_models=decision_models,
                                ens_models=ens_models,
                                aux_samples=aux_samples) 
                    save_one_img(adv_feature.detach().cpu(), adv_output_file)

                    logits = 0.0 
                    for md in ens_models:
                        outputs = md(adv_feature)
                        logits = logits + outputs
                    logits = logits / len(ens_models)

                    adv_confidence = F.softmax(logits, dim=1)
                    adv_confidences[fname_basename] = adv_confidence.detach().cpu().numpy()

                adv_output_confidence = os.path.join(adv_output_dir, 'confidence.npy')
                with open(adv_output_confidence, 'wb') as fp:
                    pickle.dump(adv_confidences, fp)

if __name__ == '__main__':
    main()
