#!/usr/bin/env python
# coding=utf-8
from torchvision.models import resnet50, vgg19, inception_v3, densenet121, wide_resnet50_2
from torchvision import transforms
import timm
import copy
import torch.nn as nn
import torch
import pdb

class ModelZoo:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.res50_feature_layers = list(range(4, 21))
        self.dense121_feature_layers = list(range(4, 12))
        self.vgg19_feature_layers = [2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 37]
        self.incep3_feature_layers = [1] 
        # self.incep3_feature_layers = list(range(1, 20))

        self.res50_saliency_layer = 'layer4'
        self.dense121_saliency_layer = 'features'
        self.incep3_saliency_layer = 'Mixed_7c'
        self.vgg19_saliency_layer = 'features'

    def get_saliency_layer(self, model_name):
        if model_name == 'resnet50' or model_name == 'wide_resnet50_2':
            return self.res50_saliency_layer
        elif model_name == 'vgg19':
            return self.vgg19_saliency_layer
        elif model_name == 'densenet121':
            return self.dense121_saliency_layer
        elif model_name == 'inception_v3':
            return self.incep3_saliency_layer
        else:
            raise 'Invalid model name!!!'


    def default_split(self, model_name, split_index=-1):
        if model_name == 'resnet50':
            res50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            children = self._get_resnet_children(res50)
            feature_model = torch.nn.Sequential(*children[:self.res50_feature_layers[split_index]])
            decision_model = torch.nn.Sequential(*children[self.res50_feature_layers[split_index]:])
        elif model_name == 'vgg19':
            v19 = vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
            children = self._get_vgg_children(v19)
            feature_model = torch.nn.Sequential(*children[:self.vgg19_feature_layers[split_index]])
            decision_model = torch.nn.Sequential(*children[self.vgg19_feature_layers[split_index]:])
        elif model_name == 'inception_v3':
            # aux_logits is False: 3x224x224 can work
            # or 
            # spliting should eliminate InceptionAux, however performance degrades
            incep3 = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
            children = self._get_inception_children(incep3)

            '''
            nchildren = []
            for c in children:
                if c.__class__.__name__ != 'InceptionAux':
                    nchildren.append(c)
            feature_model = torch.nn.Sequential(*nchildren[:self.incep3_feature_layers[-1]])
            decision_model = torch.nn.Sequential(*nchildren[self.incep3_feature_layers[-1]:])
            '''
            feature_model = torch.nn.Sequential(*children[:self.incep3_feature_layers[split_index]])
            decision_model = torch.nn.Sequential(*children[self.incep3_feature_layers[split_index]:])
        elif model_name == 'densenet121':
            dense121 = densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
            children = self._get_densenet_children(dense121)
            feature_model = torch.nn.Sequential(*children[:self.dense121_feature_layers[split_index]])
            decision_model = torch.nn.Sequential(*children[self.dense121_feature_layers[split_index]:])
        elif model_name == 'wide_resnet50_2':
            wres50 = wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1')
            children = self._get_resnet_children(wres50)
            feature_model = torch.nn.Sequential(*children[:self.res50_feature_layers[split_index]])
            decision_model = torch.nn.Sequential(*children[self.res50_feature_layers[split_index]:])
        elif model_name in ['inception_v4', 'inception_resnet_v2', 'ens_adv_inception_resnet_v2', 'adv_inception_v3']:
            return None, None
        else:
            raise 'Invalid model name!!!'

        feature_model = torch.nn.Sequential(
            transforms.Normalize(mean=self.mean, std=self.std),
            feature_model
        )
        return feature_model, decision_model

    def ushape_split(self, model_name, split_indexes=[-2, -1]):
        if model_name == 'resnet50':
            res50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            children = self._get_resnet_children(res50)

            feature_model1 = torch.nn.Sequential(*children[:self.res50_feature_layers[split_indexes[0]]])
            feature_model2 = torch.nn.Sequential(*children[self.res50_feature_layers[split_indexes[0]]:self.res50_feature_layers[split_indexes[1]]])
            decision_model = torch.nn.Sequential(*children[self.res50_feature_layers[split_indexes[1]]:])
        elif model_name == 'vgg19':
            v19 = vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
            children = self._get_vgg_children(v19)

            feature_model1 = torch.nn.Sequential(*children[:self.vgg19_feature_layers[split_indexes[0]]])
            feature_model2 = torch.nn.Sequential(*children[self.vgg19_feature_layers[split_indexes[0]]:self.vgg19_feature_layers[split_indexes[1]]])
            decision_model = torch.nn.Sequential(*children[self.vgg19_feature_layers[split_indexes[1]]:])
        elif model_name == 'inception_v3':
            # aux_logits is False: 3x224x224 can work
            # or 
            # spliting should eliminate InceptionAux, however performance degrades
            incep3 = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
            children = self._get_inception_children(incep3)

            feature_model1 = torch.nn.Sequential(*children[:self.incep3_feature_layers[split_indexes[0]]])
            feature_model2 = torch.nn.Sequential(*children[self.incep3_feature_layers[split_indexes[0]]:self.incep3_feature_layers[split_indexes[1]]])
            decision_model = torch.nn.Sequential(*children[self.incep3_feature_layers[split_indexes[1]]:])
        elif model_name == 'densenet121':
            dense121 = densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
            children = self._get_densenet_children(dense121)

            feature_model1 = torch.nn.Sequential(*children[:self.dense121_feature_layers[split_indexes[0]]])
            feature_model2 = torch.nn.Sequential(*children[self.dense121_feature_layers[split_indexes[0]]:self.dense121_feature_layers[split_indexes[1]]])
            decision_model = torch.nn.Sequential(*children[self.dense121_feature_layers[split_indexes[1]]:])
        elif model_name == 'wide_resnet50_2':
            wres50 = wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1')
            children = self._get_resnet_children(wres50)

            feature_model1 = torch.nn.Sequential(*children[:self.res50_feature_layers[split_indexes[0]]])
            feature_model2 = torch.nn.Sequential(*children[self.res50_feature_layers[split_indexes[0]]:self.res50_feature_layers[split_indexes[1]]])
            decision_model = torch.nn.Sequential(*children[self.res50_feature_layers[split_indexes[1]]:])
        elif model_name in ['inception_v4', 'inception_resnet_v2', 'ens_adv_inception_resnet_v2', 'adv_inception_v3']:
            return None, None, None
        else:
            raise 'Invalid model name!!!'

        feature_model1 = torch.nn.Sequential(
            transforms.Normalize(mean=self.mean, std=self.std),
            feature_model1
        )
        return feature_model1, feature_model2, decision_model

    def _refine_children(self, children):
        nchildren = []
        for c in children:
            if isinstance(c, torch.nn.Sequential):
                nchildren.extend(list(c.children()))
            else:
                nchildren.append(c)
        return nchildren

    def _get_resnet_children(self, model):
        children = list(model.children())
        children = self._refine_children(children)
        children.insert(-1, torch.nn.Flatten())

        return children
    
    def _get_inception_children(self, model):
        # input size should be 3 x 299 x 299
        feature_model = copy.deepcopy(model)
        feature_model.avgpool = nn.Identity()
        feature_model.dropout = nn.Identity()
        feature_model.ffc = nn.Identity()

        children = list(model.children())
        children.insert(-1, torch.nn.Flatten())
        decision_model = nn.Sequential(*children[-4:])
        return [feature_model, decision_model] 
        # return children
    
    def _get_densenet_children(self, model):
        children = list(model.children())
        children = list(children[0].children()) + children[1:]
        children.insert(-1, torch.nn.ReLU(inplace=True))
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        return children

    def _get_goolenet_children(self, model):
        children = list(model.children())
        children.insert(-2, torch.nn.Flatten())
        return children

    def _get_mobilenet_children(self, model):
        children = list(model.children())
        children = self._refine_children(children)
        children.insert(-2, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        return children

    def _get_shufflenet_children(self, model):
        children = list(model.children())
        children = self._refine_children(children)
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        return children

    def _get_squeezenet_children(self, model):
        children = list(model.children())
        children = self._refine_children(children)
        children.append(torch.nn.Flatten())
        return children

    def _get_vgg_children(self, model):
        children = list(model.children())
        children = self._refine_children(children)
        children.insert(-7, torch.nn.Flatten())
        return children

    def pick_model_pool(self, model_names):
        model_pool = []
        for mname in model_names:
            model_pool.append(self.pick_model(mname))

        return model_pool

    def pick_model(self, model_name):
        if model_name == 'resnet50':
            res50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            res50.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                res50
            )
        elif model_name == 'vgg19':
            v19 = vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
            v19.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                v19
            )
        elif model_name == 'inception_v3':
            # aux_logits is False: 3x224x224 can work
            # or eval mode
            # incep3 = inception_v3(pretrained=True, aux_logits=False)
            incep3 = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
            incep3.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                incep3
            )
        elif model_name == 'densenet121':
            dense121 = densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
            dense121.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                dense121
            )
        elif model_name == 'wide_resnet50_2':
            wres50 = wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1')
            wres50.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                wres50
            )
        elif model_name == 'inception_v4':
            incep4 = timm.create_model('inception_v4', pretrained=True)
            incep4.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                incep4 
            )
        elif model_name == 'inception_resnet_v2':
            incres_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
            incres_v2.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                incres_v2 
            )
        elif model_name == 'ens_adv_inception_resnet_v2':
            incres_v2_adv = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
            incres_v2_adv.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                incres_v2_adv
            )
        elif model_name == 'adv_inception_v3':
            incep3_adv = timm.create_model('adv_inception_v3', pretrained=True)
            incep3_adv.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                incep3_adv
            )
        else:
            raise 'Invalid model name!!!'


if __name__ == '__main__':
    mz = ModelZoo()
    # model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
    model_names = ['inception_v3']
    # model_names = ['inception_v3', 'inception_v4', 'inception_resnet_v2', 'ens_adv_inception_resnet_v2', 'adv_inception_v3']
    for mn in model_names:
        print(mn)
        # print(mz.pick_model(mn))
        # pdb.set_trace()
        feature_model, decision_model = mz.default_split(mn, split_index=-1)
        # print(feature_model)
        # print(decision_model)
        # model = mz.pick_model(mn)
        pdb.set_trace()
        print('debug')
