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

    # for mname in model_names:
    model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
    # ens_names = ['resnet50', 'densenet121', 'wide_resnet50_2']
    # vic_names = ['resnet50', 'densenet121', 'wide_resnet50_2']
    # vic_names = ['resnet50', 'wide_resnet50_2']
    # vic_names = ['vgg19', 'inception_v3']
    vic_names = ['inception_resnet_v2']
    # for mname in ['vgg19', 'inception_v3']:
    for mname in vic_names: 
        print('model {} generates adversarial examples...'.format(mname))

        adv_output_dir = os.path.join(ad_output_path, mname)
        if not os.path.exists(adv_output_dir):
            os.makedirs(adv_output_dir)

        for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):

            adv_output_dir = os.path.join(ad_output_path, mname, dname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)

            print('1. dataset {} is attacked...'.format(dname)) 
            ds = dzoo.load_dataset(dname, dpath)
            label_space = list(ds.class_to_idx.values())

            holdout_model = mzoo.pick_model(mname)
            holdout_model.eval()
            holdout_model = holdout_model.to(device)

            ens_models = np.setdiff1d(model_names, [mname])
            ens_models = mzoo.pick_model_pool(ens_models)
            for md in ens_models:
                md.eval()
            ens_models = [md.to(device) for md in ens_models]

            for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                random_seed()

                # skip some methods
                # if attack_name in ['MI-FGSM', 'NI-FGSM', 'AA', 'TAA', 'BSI-FGSM', 'VMI-FGSM', 'VNI-FGSM', 'BAP', 'SINI-FGSM', 'DI-FGSM']:
                # if attack_name in ['MI-FGSM', 'NI-FGSM', 'AA', 'TAA', 'BSI-FGSM', 'VMI-FGSM', 'DI-FGSM']:
                # if attack_name in ['AA', 'TAA', 'BSI-FGSM', 'VNI-FGSM', 'BAP', 'SINI-FGSM', 'DI-FGSM']:
                if attack_name in ['AA', 'TAA', 'BSI-FGSM']:
                    continue

                # if attack_name not in ['NI-FGSM', 'DDNI_FGSM']:
                # if attack_name not in ['DDMI-FGSM']:
                # if attack_name not in ['DDNI-FGSM']:
                # if attack_name not in ['DDSINI-FGSM', 'SINI-FGSM']:
                # if attack_name not in ['DDSINI-FGSM', 'SINI-FGSM']:
                # if attack_name not in ['DISINI-FGSM']:
                # if attack_name not in ['ENSBAP', 'ENSSIT', 'ENSMI-FGSM', 'ENSDI-FGSM', 'ENSTI-FGSM', 'ENSVMI-FGSM', 'ENSVNI-FGSM', 'ENSPoincare']: # for vgg19, inc-v3
                # if attack_name not in ['ENSMI-FGSM', 'ENSDI-FGSM', 'ENSTI-FGSM', 'ENSPoincare']: # for res50, wres50, dn121
                if attack_name not in ['ENSBAP', 'ENSSIT']:
                # if attack_name not in ['ENSSIT']:
                # if attack_name not in ['ENSTI-FGSM', 'ENSSIT', 'ENSPoincare']:
                # if attack_name not in ['ENSVMI-FGSM', 'ENSVNI-FGSM']:
                    continue

                if attack_name in ['AA', 'TAA']:
                    feature_library = {}
                    print('1.* building feature library for {}...'.format(dname))

                    fb_ds = dzoo.load_dataset(fbname, os.path.join(fbpath, mname))
                    k = 0
                    debug = False 
                    for (fs, lab) in tqdm(fb_ds):
                        if lab in feature_library:
                            feature_library[lab].append(fs.unsqueeze(0))
                        else:
                            feature_library[lab] = [fs.unsqueeze(0)]

                        k = k + 1
                        if k > 100 and debug:
                            break
                    for lab in feature_library.keys():
                        feature_library[lab] = torch.cat(feature_library[lab])
                elif attack_name in ['BSI-FGSM']:
                    saliency_layer_name = mzoo.get_saliency_layer(mname)
                    saliency_layer = getattr(model[1], saliency_layer_name)

                adv_output_dir = os.path.join(ad_output_path, mname, dname, attack_name)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('2.{} attack method {} is attacking...'.format(i, attack_name))
                if attack_name == 'ENSDI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_difgsm.ENSDIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            diversity_prob=attack_args['diversity_prob'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSPoincare':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_poincare.ENSPoincare(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            diversity_prob=attack_args['diversity_prob'],
                            lamb=attack_args['lamb'],
                            margin=attack_args['margin'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSMI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_mifgsm.ENSMIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha)
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSVMI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_vmifgsm.ENSVMIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha)
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSVNI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_vnifgsm.ENSVNIFGSM(holdout_model, 
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha)
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSBAP':
                    alpha = 1.0 
                    attack = attacks.ens_bap.ENSBAP(holdout_model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n=10, mixup_num=4, mixup_weight=0.4)
                    #        n=attack_args['n'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSTI-FGSM':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_tifgsm.ENSTIFGSM(holdout_model, eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            decay=attack_args['decay_factor'],
                            alpha=alpha,
                            len_kernel=attack_args['len_kernel'],
                            diversity_prob=attack_args['diversity_prob'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSSIT':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_sit.ENSSIT(holdout_model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n_copies=10)
                    #        n=attack_args['n'])
                    # targeted
                    attack.set_mode_targeted_by_label()
                elif attack_name == 'ENSSSA':
                    alpha = attack_args['eps'] / attack_args['max_iter']
                    attack = attacks.ens_ssa.ENSSSA(holdout_model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n_copies=10)
                    # targeted
                    attack.set_mode_targeted_by_label()
                else:
                    # raise 'Invalid attack method!!!'
                    continue

                # begin to attack
                adv_confidences = {} 
                for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                    feature = feature.unsqueeze(0).to(device)
                    source = torch.LongTensor([label]).to(device)

                    fname_basename = os.path.basename(fname)
                    (_, target) = attack_targets[fname_basename]
                    target = torch.LongTensor([target]).to(device)
                    adv_output_file = os.path.join(adv_output_dir, fname_basename)

                    adv_feature = attack(feature, target, source_labels=source, ens_models=ens_models) 
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
