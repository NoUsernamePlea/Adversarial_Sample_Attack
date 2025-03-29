#!/o0usr/bin/env python
# coding=utf-8
# model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
model_names = ['resnet50', 'densenet121', 'wide_resnet50_2', 'vgg19']
# model_names = ['inception_v3', 'inception_v4', 'inception_resnet_v2', 'ens_adv_inception_resnet_v2', 'adv_inception_v3', 'resnet50', 'vgg19', 'densenet121', 'wide_resnet50_2']
# victim_model_names = ['inception_v3', 'inception_v4', 'inception_resnet_v2', 'ens_adv_inception_resnet_v2', 'adv_inception_v3', 'resnet50', 'vgg19', 'densenet121', 'wide_resnet50_2']
victim_model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
# victim_datasets = [('imagenet', '/home/zero/zero/split_dp/dataset/imagenet/new_adv_15k')]
victim_datasets = [('imagenet', '/home/zero/zero/split_dp/dataset/imagenet/new_adv_1k')]
feature_libraries = [('imagenet', '/home/zero/zero/split_dp/dataset/imagenet/feature_library')]
clean_libraries = ('imagenet', '/home/zero/zero/split_dp/dataset/imagenet/train_by_class')
adv_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dadv_outputs'
lr_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dlr_outputs'
n_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dn_outputs'
layer_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dlayer_outputs'
mlayer_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dmlayer_outputs'
mixup_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dmixup_outputs'
kern_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dkern_outputs'
eps_output_path = '/home/zero/zero/split_dp/dataset/imagenet/deps_outputs'
steps_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dsteps_outputs'
ens_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dens_outputs'
lint_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dlint_outputs'
ad_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dad_outputs'
hgd_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dhgd_outputs'
test_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dtest_outputs'
region_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dregion_outputs'
dd_output_path = '/home/zero/zero/split_dp/dataset/imagenet/ddd_outputs'
gamma_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dgamma_outputs'
mo_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dmo_outputs'
smo_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dsmo_outputs'
fix_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dfix_outputs'
cfix_output_path = '/home/zero/zero/split_dp/dataset/imagenet/tcfix_outputs'
beta1_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dbeta1_outputs'
fbeta1_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dfbeta1_outputs'
beta2_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dbeta2_outputs'
# attack_book = './attack_book_15k.json'
attack_book = './new_attack_book_1k.json'

proposed_attack_methods = {
    'N2A': {
        'max_iter': 10,            # iterations
        'lr': 0.25,
        'pos_lr': 0.01,
        'eps': 0.07,    # perturbation
        'feature_model': True,
        'c1': 1.0,
        'c2': 1.0,
        'c3': 0.1,
        'threshold': 0.5,
        'candidates': 1,
        'mask_args': {
            'lr': 0.05,
            'epochs': 50,
            'c': 10.0,
            'm': 1 
        }
    },
}

proposed_evaluation_file = 'evaluation/dproposed_evaluation.csv'
baseline_evaluation_file = 'evaluation/dbaseline_evaluation.csv'
ens_evaluation_file = 'evaluation/dens_evaluation.csv'
ad_evaluation_file = 'evaluation/dad_evaluation.csv'
hgd_evaluation_file = 'evaluation/dhgd_evaluation.csv'
trans_evaluation_file = 'evaluation/dtrans_evaluation.csv'
rp_evaluation_file = 'evaluation/drp_evaluation.csv'
fd_evaluation_file = 'evaluation/dfd_evaluation.csv'
fs_evaluation_file = 'evaluation/dfs_evaluation.csv'
nrp_evaluation_file = 'evaluation/dnrp_evaluation.csv'
lr_evaluation_file = 'evaluation/dlr_evaluation.csv'
eps_evaluation_file = 'evaluation/deps_evaluation.csv'
n_evaluation_file = 'evaluation/dn_evaluation.csv'
lint_evaluation_file = 'evaluation/dlint_evaluation.csv'
layer_evaluation_file = 'evaluation/dlayer_evaluation.csv'
mlayer_evaluation_file = 'evaluation/dmlayer_evaluation.csv'
steps_evaluation_file = 'evaluation/dsteps_evaluation.csv'
test_evaluation_file = 'evaluation/dtest_evaluation.csv'
mixup_evaluation_file = 'evaluation/dmixup_evaluation.csv'
kern_evaluation_file = 'evaluation/dkern_evaluation.csv'
region_evaluation_file = 'evaluation/dregion_evaluation.csv'
dd_evaluation_file = 'evaluation/ddd_evaluation.csv'
gamma_evaluation_file = 'evaluation/dgamma_evaluation.csv'
mo_evaluation_file = 'evaluation/dmo_evaluation.csv'
smo_evaluation_file = 'evaluation/dsmo_evaluation.csv'
fix_evaluation_file = 'evaluation/dfix_evaluation.csv'
cfix_evaluation_file = 'evaluation/dcfix_evaluation.csv'
beta1_evaluation_file = 'evaluation/dbeta1_evaluation.csv'
fbeta1_evaluation_file = 'evaluation/dfbeta1_evaluation.csv'
beta2_evaluation_file = 'evaluation/dbeta2_evaluation.csv'
baseline_attack_methods = {
    'DI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'LINTDI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'DDDI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'TI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'len_kernel': 7,
        'feature_model': False,
    },
    'LINTTI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'len_kernel': 7,
        'feature_model': False,
    },
    'ENSTI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'len_kernel': 7,
        'feature_model': False,
    },
    'GTI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'len_kernel': 7,
        'feature_model': False,
    },
    'DDTI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'len_kernel': 7,
        'feature_model': False,
    },
    'Poincare': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'lamb': 0.01,
        'margin': 0.007,
        'feature_model': False,
    },
    'ENSPoincare': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'lamb': 0.01,
        'margin': 0.007,
        'feature_model': False,
    },
    'Logit': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'lamb': 0.01,
        'margin': 0.007,
        'feature_model': False,
    },
    'MI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'CFM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'NI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'SINI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'VMI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'decay_factor': 1.0,          # decay factor
        'feature_model': False,
    },
    'VNI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'decay_factor': 1.0,          # decay factor
        'feature_model': False,
    },
#    'AA': {
#        'max_iter': 10,            # iterations
#        'eps': 0.07,    # perturbation
#        'feature_model': True,
#    },
#    'TAA': {
#        'max_iter': 10,            # iterations
#        'eps': 0.07,    # perturbation
#        'feature_model': True,
#    },
    'BAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step sizes 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'PBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True, #LBAP
    },
    'LBAP-MMix': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True, #LBAP-MMix
    },
    'LBAP-MConv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True, #LBAP-MConv
    },
    'LBAP-MMixConv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True, # LBAP-MMixConv
    },
    'LBAP-MConvMix': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True, # LBAP-MConvMix
    },
    'LBAP-parallel': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-mixconv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-convmix': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-convconv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-mixmix': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-concatenate': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LBAP-Conv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'ENSLBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LLBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'APBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'ENSAPBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'MPBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'MBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'SIT': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LINTSIT': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'LINTSIT-Conv': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'DDSIT': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'TSIT': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'SSA': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'ENSDI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'ENSMI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'ENSSIT': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'ENSNI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'ENSSINI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'ENSBAP': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
        'feature_model': True,
    },
    'DDPoincare': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'lamb': 0.01,
        'n': 10,
        'margin': 0.007,
        'feature_model': False,
    },
    'DISINI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'DDSINI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'DISINI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'DDNI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'n': 10,
        'feature_model': False,
    },
    'DIMI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'DDMI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'n': 10,
        'feature_model': False,
    },
    'ENSVMI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'decay_factor': 1.0,          # decay factor
        'feature_model': False,
    },
    'ENSVNI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'decay_factor': 1.0,          # decay factor
        'feature_model': False,
    },
    'Admix': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'ratio': 0.2,
        'n': 3,
        'feature_model': False,
    },
    'LINTAdmix': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'ratio': 0.2,
        'n': 3,
        'feature_model': False,
    },
    'LINTAdmix-Conv': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'ratio': 0.2,
        'n': 3,
        'feature_model': False,
    },
    'DEM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.069,    # perturbation
        'feature_model': False,
    },
    'LINTDEM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'LINTDEM-Conv': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
    'ENSAdmix': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'ratio': 0.2,
        'n': 3,
        'feature_model': False,
     },
    'ENSAdmix': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'ratio': 0.2,
        'n': 3,
        'feature_model': False,
    },
    'ENSDEM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'feature_model': False,
    },
}

